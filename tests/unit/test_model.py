# tests/unit/test_model.py
import numpy as np
import pandas as pd
import pytest
from src.model.train import (
    add_match_importance,
    augment_neutral_matches,
    baseline_accuracies,
    DIFF_FEATURES,
    evaluate_on_world_cup,
    FEATURE_COLUMNS,
)


def _diff_cols(value):
    return {col: value for col in [
        "days_since_last_match_diff", "win_streak_diff", "unbeaten_streak_diff",
        "h2h_matches_played_diff", "h2h_win_rate_diff",
        "form_points_avg_5_diff", "form_goals_for_avg_5_diff",
        "form_goals_against_avg_5_diff", "form_win_rate_5_diff",
        "form_points_avg_10_diff", "form_goals_for_avg_10_diff",
        "form_goals_against_avg_10_diff", "form_win_rate_10_diff",
    ]}


@pytest.fixture
def league_history():
    """
    Six Brazil vs Argentina friendlies through 2021, followed by a
    'World Cup 2022' match — enough to exercise a real train/test split
    without needing the full engineered-features pipeline.
    """
    rows = []
    for i, date in enumerate(pd.date_range("2021-01-01", periods=6, freq="30D")):
        rows.append({
            "date": date,
            "home_team": "Brazil", "away_team": "Argentina",
            "tournament": "Friendly", "neutral": False,
            "is_world_cup": 0,
            "elo_diff": 50.0, "outcome": 2,
            **_diff_cols(0.1 * i),
        })
    rows.append({
        "date": pd.Timestamp("2022-11-24"),
        "home_team": "Brazil", "away_team": "Argentina",
        "tournament": "FIFA World Cup", "neutral": True,
        "is_world_cup": 1,
        "elo_diff": 60.0, "outcome": 0,
        **_diff_cols(0.0),
    })
    return pd.DataFrame(rows)


def test_add_match_importance_matches_elo_k_factors(league_history):
    """Reuses elo.py's K-factor table — World Cup should score higher than Friendly."""
    result = add_match_importance(league_history)
    wc_importance = result[result["tournament"] == "FIFA World Cup"]["match_importance"].iloc[0]
    friendly_importance = result[result["tournament"] == "Friendly"]["match_importance"].iloc[0]
    assert wc_importance > friendly_importance
    assert wc_importance == 60.0
    assert friendly_importance == 20.0


def test_evaluate_on_world_cup_excludes_target_year_from_training(league_history, monkeypatch):
    """
    The 2022 holdout must never train on the 2022 match itself —
    patch train_xgb to record which years it actually saw.
    """
    import src.model.train as train_mod

    class _StubModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            proba = np.zeros((len(X), 3))
            proba[:, 0] = 1.0
            return proba

    seen_train_rows = {}

    def fake_train_xgb(X_train, y_train):
        seen_train_rows["index"] = X_train.index
        return _StubModel()

    monkeypatch.setattr(train_mod, "train_xgb", fake_train_xgb)

    df = add_match_importance(league_history)
    evaluate_on_world_cup(df, wc_year=2022)

    train_years = df.loc[seen_train_rows["index"], "date"].dt.year.unique().tolist()
    assert 2022 not in train_years


def test_baseline_accuracies_elo_favorite_perfect_when_elo_always_right():
    """If elo_diff sign always matches the outcome, elo_favorite_acc should be 1.0."""
    test_df = pd.DataFrame({
        "outcome":  [2, 0, 2, 0],
        "elo_diff": [100, -50, 30, -10],
    })
    result = baseline_accuracies(test_df)
    assert result["elo_favorite_acc"] == pytest.approx(1.0)


def test_baseline_accuracies_majority_class():
    """majority_class_acc should equal the frequency of the most common outcome."""
    test_df = pd.DataFrame({
        "outcome":  [2, 2, 2, 0, 1],
        "elo_diff": [0, 0, 0, 0, 0],
    })
    result = baseline_accuracies(test_df)
    assert result["majority_class_acc"] == pytest.approx(3 / 5)


def test_feature_columns_exclude_leakage():
    """Post-match columns must never appear in the feature set — that would leak the outcome."""
    leakage_columns = {
        "home_score", "away_score", "outcome", "goal_diff", "total_goals",
        "home_elo_after", "away_elo_after",
    }
    assert leakage_columns.isdisjoint(FEATURE_COLUMNS)


@pytest.fixture
def mixed_neutral_df():
    """
    Two neutral rows and one non-neutral row, distinguishable by elo_diff/outcome.
    Diff features use a non-zero value (0.1, 0.2, 0.3) — negating 0.0 would
    trivially "pass" a broken sign-flip, so 0.0 can't catch that bug.
    """
    return pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
        "home_team": ["Brazil", "France", "Spain"],
        "away_team": ["Argentina", "Germany", "Italy"],
        "tournament": ["FIFA World Cup", "FIFA World Cup", "Friendly"],
        "neutral": [True, True, False],
        "is_world_cup": [1, 1, 0],
        "match_importance": [60.0, 60.0, 20.0],
        "outcome": [2, 1, 0],  # home win, draw, away win
        "elo_diff": [80.0, 40.0, -40.0],
        **{col: [0.1, 0.2, 0.3] for col in [
            "days_since_last_match_diff", "win_streak_diff", "unbeaten_streak_diff",
            "h2h_matches_played_diff", "h2h_win_rate_diff",
            "form_points_avg_5_diff", "form_goals_for_avg_5_diff",
            "form_goals_against_avg_5_diff", "form_win_rate_5_diff",
            "form_points_avg_10_diff", "form_goals_for_avg_10_diff",
            "form_goals_against_avg_10_diff", "form_win_rate_10_diff",
        ]},
    })


def test_augment_neutral_matches_doubles_only_neutral_rows(mixed_neutral_df):
    result = augment_neutral_matches(mixed_neutral_df)
    assert len(result) == len(mixed_neutral_df) + 2  # only the 2 neutral rows get mirrored
    assert (result["neutral"] == False).sum() == 1  # noqa: E712 — non-neutral row untouched


def test_augment_neutral_matches_negates_diff_features(mixed_neutral_df):
    result = augment_neutral_matches(mixed_neutral_df)
    mirrored = result.iloc[len(mixed_neutral_df):]  # the appended rows
    original_brazil_row = mixed_neutral_df.iloc[0]
    mirrored_row = mirrored[
        (mirrored["home_team"] == "Argentina") & (mirrored["away_team"] == "Brazil")
    ].iloc[0]
    for col in DIFF_FEATURES:
        assert mirrored_row[col] == pytest.approx(-original_brazil_row[col])


def test_augment_neutral_matches_flips_outcome_preserves_draw(mixed_neutral_df):
    result = augment_neutral_matches(mixed_neutral_df)
    mirrored = result.iloc[len(mixed_neutral_df):]

    home_win_mirror = mirrored[mirrored["away_team"] == "Brazil"].iloc[0]  # was home win (2)
    draw_mirror = mirrored[mirrored["away_team"] == "France"].iloc[0]      # was draw (1)
    assert home_win_mirror["outcome"] == 0
    assert draw_mirror["outcome"] == 1


def test_augment_neutral_matches_preserves_match_context(mixed_neutral_df):
    """Mirrored rows keep the same neutral/is_world_cup/match_importance — only diffs/outcome/teams change."""
    result = augment_neutral_matches(mixed_neutral_df)
    mirrored_row = result.iloc[len(mixed_neutral_df):].iloc[0]
    assert mirrored_row["neutral"] == True  # noqa: E712
    assert mirrored_row["is_world_cup"] == 1
    assert mirrored_row["match_importance"] == 60.0


def test_augment_neutral_matches_noop_when_nothing_neutral():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01"]),
        "home_team": ["Brazil"], "away_team": ["Argentina"],
        "neutral": [False], "is_world_cup": [0], "match_importance": [20.0],
        "outcome": [2], "elo_diff": [50.0],
        **_diff_cols(0.0),
    })
    result = augment_neutral_matches(df)
    assert len(result) == len(df)
