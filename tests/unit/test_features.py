# tests/unit/test_features.py
import pandas as pd
import pytest
from src.features.engineering import build_features


@pytest.fixture
def matches():
    """
    Brazil vs Argentina, alternating venues, six matches spanning two years.
    Chosen so each team has both wins and losses, and meets the same
    opponent every time — enough to exercise form, streaks, and H2H.
    """
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-04-01", "2020-05-01", "2020-06-01",
        ]),
        "home_team":   ["Brazil", "Argentina", "Brazil", "Argentina", "Brazil", "Argentina"],
        "away_team":   ["Argentina", "Brazil", "Argentina", "Brazil", "Argentina", "Brazil"],
        "home_score":  [2, 0, 1, 1, 3, 0],
        "away_score":  [0, 1, 1, 0, 0, 2],
        "tournament":  ["Friendly"] * 6,
        "neutral":     [False] * 6,
        "is_world_cup": [0] * 6,
    })


def test_first_appearance_has_no_history(matches):
    """A team's very first match should have NaN form/streak/H2H features — no lookahead, no fabricated history."""
    result = build_features(matches)
    first = result.iloc[0]
    assert pd.isna(first["home_form_points_avg_5"])
    assert pd.isna(first["home_h2h_win_rate"])
    assert first["home_win_streak"] == 0
    assert first["home_h2h_matches_played"] == 0


def test_no_lookahead_in_form(matches):
    """
    Row 2 (Argentina 0-1 Brazil, 2020-02-01): Brazil is away here after
    winning 2-0 as home team in row 0. Brazil's away_form_win_rate_5
    going into this match must reflect only that one prior win (1.0),
    never the outcome of this match itself.
    """
    result = build_features(matches)
    row = result.iloc[1]
    assert row["away_team"] == "Brazil"
    assert row["away_form_win_rate_5"] == pytest.approx(1.0)
    assert row["away_win_streak"] == 1


def test_head_to_head_accumulates(matches):
    """By the last match, each team has met the other 5 times before — h2h_matches_played must count only prior meetings."""
    result = build_features(matches)
    last = result.iloc[-1]
    assert last["home_h2h_matches_played"] == 5
    assert last["away_h2h_matches_played"] == 5


def test_win_streak_resets_on_loss(matches):
    """
    Brazil: W (row0) D... actually loses as away in row1 (0-1), so its
    streak going into row2 (home vs Argentina) should reset to 0 wins,
    but still count as unbeaten only through the win — check reset explicitly
    using a small forced sequence.
    """
    losers = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-15"]),
        "home_team":  ["France", "France", "France"],
        "away_team":  ["Spain", "Spain", "Spain"],
        "home_score": [2, 0, 1],
        "away_score": [0, 1, 1],
        "tournament": ["Friendly"] * 3,
        "neutral":    [False] * 3,
        "is_world_cup": [0] * 3,
    })
    result = build_features(losers)
    assert result.iloc[1]["home_win_streak"] == 1   # won match 1, streak going into match 2
    assert result.iloc[2]["home_win_streak"] == 0   # lost match 2, streak resets going into match 3


def test_rest_days_diff(matches):
    """home_days_since_last_match_diff should equal home minus away rest days."""
    result = build_features(matches)
    row = result.iloc[2]
    expected = row["home_days_since_last_match"] - row["away_days_since_last_match"]
    assert row["days_since_last_match_diff"] == pytest.approx(expected)


def test_output_row_order_matches_input_dates(matches):
    """Output should stay chronologically ordered, matching the input."""
    result = build_features(matches)
    assert result["date"].is_monotonic_increasing


def test_original_columns_preserved(matches):
    """Feature engineering should not drop any original match columns."""
    result = build_features(matches)
    for col in matches.columns:
        assert col in result.columns
