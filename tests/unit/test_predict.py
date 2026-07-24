# tests/unit/test_predict.py
import numpy as np
import pandas as pd
import pytest
from src.data.elo import INITIAL_RATING, calculate_elo
from src.model.predict import (
    _elo_as_of,
    _is_world_cup,
    get_all_teams,
    get_current_team_stats,
    get_head_to_head,
    predict_latest,
    predict_match,
    predict_with_model,
    train_as_of,
)
from src.model.train import FEATURE_COLUMNS, FEATURE_LABELS


@pytest.fixture
def matches():
    """
    ~30 Brazil/Argentina/France friendlies plus a couple of World Cup matches,
    run through the real calculate_elo() so elo_before/after are internally
    consistent — enough history for build_features' rolling windows to have
    real (non-NaN) values by the cutoff date used in the tests below.
    """
    home_score = [2, 1, 0] * 10
    away_score = [1, 1, 1] * 10
    raw = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=30, freq="20D"),
        "home_team": (["Brazil", "Argentina", "France"] * 10),
        "away_team": (["Argentina", "France", "Brazil"] * 10),
        "home_score": home_score,
        "away_score": away_score,
        "tournament": (["Friendly"] * 27 + ["FIFA World Cup"] * 3),
        "neutral": [False] * 30,
        "is_world_cup": ([0] * 27 + [1] * 3),
        # Same encoding as src/data/clean.py: 2=home win, 1=draw, 0=away win.
        "outcome": np.select(
            [np.array(home_score) > np.array(away_score), np.array(home_score) == np.array(away_score)],
            [2, 1], default=0,
        ),
    })
    return calculate_elo(raw)


def test_probabilities_sum_to_one(matches):
    result = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_get_head_to_head_counts_wins_correctly(matches):
    """Brazil beats Argentina 2-1 at home every one of their 10 meetings in the fixture."""
    h2h = get_head_to_head(matches, "Brazil", "Argentina")
    assert h2h["total_matches"] == 10
    assert h2h["team_a_wins"] == 10
    assert h2h["team_b_wins"] == 0
    assert h2h["draws"] == 0


def test_get_head_to_head_symmetric_regardless_of_argument_order(matches):
    """Swapping team_a/team_b should just swap which win-count is which, not change the facts."""
    ab = get_head_to_head(matches, "Brazil", "Argentina")
    ba = get_head_to_head(matches, "Argentina", "Brazil")
    assert ab["team_a_wins"] == ba["team_b_wins"]
    assert ab["team_b_wins"] == ba["team_a_wins"]
    assert ab["total_matches"] == ba["total_matches"]


def test_get_head_to_head_no_prior_meetings(matches):
    """Two teams that have never played each other in `history` should show zero matches, not error."""
    h2h = get_head_to_head(matches, "Brazil", "Wakanda")
    assert h2h["total_matches"] == 0
    assert h2h["recent_meetings"] == []


def test_get_head_to_head_recent_meetings_most_recent_first(matches):
    h2h = get_head_to_head(matches, "Brazil", "Argentina", recent_n=3)
    dates = [m["date"] for m in h2h["recent_meetings"]]
    assert dates == sorted(dates, reverse=True)
    assert len(h2h["recent_meetings"]) == 3


def test_get_head_to_head_respects_history_cutoff(matches):
    """Meetings after the as_of cutoff must not appear — history is already the caller's responsibility."""
    cutoff_history = matches[matches["date"] < pd.Timestamp("2019-06-01")]
    h2h = get_head_to_head(cutoff_history, "Brazil", "Argentina")
    assert all(m["date"] < "2019-06-01" for m in h2h["recent_meetings"])
    assert h2h["total_matches"] < 10


def test_get_current_team_stats_streaks_and_ranking(matches):
    """
    Brazil wins every one of its 20 matches in the fixture (win_streak=20).
    Argentina's most recent match was a draw preceded by a loss
    (win_streak=0, unbeaten_streak=1). Table must be ELO-ranked, so the
    highest-ELO team (Brazil, undefeated) sorts first.
    """
    stats = get_current_team_stats(matches)

    assert list(stats["current_elo"]) == sorted(stats["current_elo"], reverse=True)

    brazil = stats[stats["team"] == "Brazil"].iloc[0]
    assert brazil["win_streak"] == 20
    assert brazil["unbeaten_streak"] == 20
    assert stats.iloc[0]["team"] == "Brazil"

    argentina = stats[stats["team"] == "Argentina"].iloc[0]
    assert argentina["win_streak"] == 0
    assert argentina["unbeaten_streak"] == 1


def test_feature_contributions_cover_every_feature_with_readable_labels(matches):
    result = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    contributions = result["feature_contributions"]

    assert len(contributions) == len(FEATURE_COLUMNS)
    labels = {c["feature"] for c in contributions}
    assert labels == set(FEATURE_LABELS.values())
    # Human-readable, not raw column names like "form_win_rate_5_diff".
    assert "elo_diff" not in labels
    assert "ELO rating gap" in labels


def test_feature_contributions_sorted_by_absolute_impact(matches):
    result = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    magnitudes = [abs(c["contribution"]) for c in result["feature_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_feature_contributions_include_raw_feature_value(matches):
    """Each contribution should carry the actual home-minus-away feature value that produced it, not just the SHAP number."""
    result = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    elo_gap = next(c for c in result["feature_contributions"] if c["feature"] == "ELO rating gap")
    assert elo_gap["value"] == pytest.approx(result["home_elo"] - result["away_elo"], abs=0.05)


def test_elo_as_of_matches_latest_prior_match(matches):
    """home_elo/away_elo in the result should equal each team's ELO right after their last match before the cutoff."""
    cutoff = pd.Timestamp("2019-06-01")
    history = matches[matches["date"] < cutoff]
    expected_brazil = _elo_as_of(history, "Brazil")

    result = predict_match("Brazil", "Argentina", cutoff, matches=matches)
    assert result["home_elo"] == pytest.approx(expected_brazil, abs=0.05)


def test_elo_as_of_excludes_the_boundary_match_itself(matches):
    """
    A team's ELO for a cutoff exactly on one of its match dates must equal
    that match's own elo_before (the rating it carried INTO the match), never
    its elo_after — otherwise the match's own result would leak into a
    same-day prediction. This is the exact property that matters for an "as
    of the day before this real match" prediction: verified by hand against
    the actual 2026 World Cup final in conversation, pinned here so it can't
    silently regress.
    """
    # Last match in the fixture — same shape as "the final" being the most
    # recent result in the real dataset.
    boundary_match = matches.iloc[-1]
    cutoff = boundary_match["date"]
    home_team = boundary_match["home_team"]

    history = matches[matches["date"] < cutoff]
    elo = _elo_as_of(history, home_team)

    assert elo == pytest.approx(boundary_match["home_elo_before"])
    assert elo != pytest.approx(boundary_match["home_elo_after"])


def test_unseen_team_uses_initial_rating(matches):
    """A team with zero match history should start at INITIAL_RATING, not error out."""
    result = predict_match("Brazil", "Wakanda", "2021-06-01", matches=matches)
    assert result["away_elo"] == pytest.approx(INITIAL_RATING)


def test_no_lookahead_future_matches_dont_change_prediction(matches):
    """
    Predicting as of 2021-06-01 must be identical whether or not the dataset
    also contains matches dated after 2021-06-01 — the whole point of "as of".
    """
    cutoff = "2021-06-01"
    matches_trimmed = matches[matches["date"] < pd.Timestamp(cutoff)]

    result_full = predict_match("Brazil", "Argentina", cutoff, matches=matches)
    result_trimmed = predict_match("Brazil", "Argentina", cutoff, matches=matches_trimmed)

    assert result_full["home_win_prob"] == pytest.approx(result_trimmed["home_win_prob"], abs=1e-6)
    assert result_full["draw_prob"] == pytest.approx(result_trimmed["draw_prob"], abs=1e-6)
    assert result_full["away_win_prob"] == pytest.approx(result_trimmed["away_win_prob"], abs=1e-6)
    assert result_full["home_elo"] == pytest.approx(result_trimmed["home_elo"])


def test_raises_when_no_prior_history(matches):
    """A cutoff before any data exists should fail loudly, not silently train on nothing."""
    with pytest.raises(ValueError):
        predict_match("Brazil", "Argentina", "2000-01-01", matches=matches)


def test_get_all_teams_is_sorted_and_deduplicated(matches):
    """Every team appears once, alphabetically — for populating a UI dropdown."""
    teams = get_all_teams(matches)
    assert teams == sorted(set(teams))
    assert "Brazil" in teams
    assert "Argentina" in teams
    assert "France" in teams


def test_is_world_cup_matches_clean_py_convention():
    assert _is_world_cup("FIFA World Cup") == 1
    assert _is_world_cup("FIFA World Cup qualification") == 0
    assert _is_world_cup("Friendly") == 0


def test_train_as_of_plus_predict_with_model_matches_predict_match(matches):
    """
    predict_match() is documented as train_as_of() + predict_with_model()
    glued together — splitting them apart must not change the answer.
    """
    cutoff = "2021-06-01"
    model, history = train_as_of(cutoff, matches=matches)
    split_result = predict_with_model(model, history, "Brazil", "Argentina", cutoff)

    one_shot_result = predict_match("Brazil", "Argentina", cutoff, matches=matches)

    assert split_result["home_win_prob"] == pytest.approx(one_shot_result["home_win_prob"])
    assert split_result["draw_prob"] == pytest.approx(one_shot_result["draw_prob"])
    assert split_result["away_win_prob"] == pytest.approx(one_shot_result["away_win_prob"])


def test_trained_model_reused_across_multiple_matchups(matches):
    """The whole point of the split: one train_as_of() call, many cheap predict_with_model() calls."""
    cutoff = "2021-06-01"
    model, history = train_as_of(cutoff, matches=matches)

    result_a = predict_with_model(model, history, "Brazil", "Argentina", cutoff)
    result_b = predict_with_model(model, history, "France", "Argentina", cutoff)

    for result in (result_a, result_b):
        total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
        assert total == pytest.approx(1.0, abs=1e-6)


def test_predict_latest_falls_back_when_no_saved_model(matches, monkeypatch, tmp_path):
    """With no production model on disk, predict_latest should train on the spot rather than error."""
    result = predict_latest("Brazil", "Argentina", matches=matches, models_dir=tmp_path)
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result["as_of_date"] == str((matches["date"].max() + pd.Timedelta(days=1)).date())


def test_is_knockout_zeroes_draw_and_splits_it_evenly(matches):
    """The draw mass should move entirely and evenly into home/away, probabilities still summing to 1."""
    without = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    with_knockout = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches, is_knockout=True)

    assert with_knockout["draw_prob"] == 0.0
    total = with_knockout["home_win_prob"] + with_knockout["draw_prob"] + with_knockout["away_win_prob"]
    assert total == pytest.approx(1.0, abs=1e-6)

    half_draw = without["draw_prob"] / 2
    assert with_knockout["home_win_prob"] == pytest.approx(without["home_win_prob"] + half_draw, abs=1e-4)
    assert with_knockout["away_win_prob"] == pytest.approx(without["away_win_prob"] + half_draw, abs=1e-4)


def test_is_knockout_is_a_noop_by_default(matches):
    """is_knockout defaults to False — existing callers must be unaffected."""
    result = predict_match("Brazil", "Argentina", "2021-06-01", matches=matches)
    assert result["draw_prob"] > 0.0


def test_predict_latest_uses_saved_model_when_available(matches, monkeypatch):
    """When a production model IS found, predict_latest must use it — not silently retrain."""
    import src.model.predict as predict_mod

    stub_model, _ = train_as_of("2021-06-01", matches=matches)
    monkeypatch.setattr(predict_mod, "load_production_model", lambda models_dir=None: stub_model)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("train_as_of should not be called when a saved model is found")

    monkeypatch.setattr(predict_mod, "train_as_of", fail_if_called)

    result = predict_latest("Brazil", "Argentina", matches=matches)
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0, abs=1e-6)
