# tests/test_elo.py
import pandas as pd
import pytest
from src.data.elo import expected_score, update_ratings, calculate_elo, INITIAL_RATING


# ── Unit tests for individual functions ──────────────────────────────────────

def test_expected_score_equal_ratings():
    """Equal ratings should give 0.5 expected score."""
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_rating_wins():
    """Higher rated team should have expected score above 0.5."""
    assert expected_score(1700, 1500) > 0.5
    assert expected_score(1500, 1700) < 0.5


def test_expected_score_symmetric():
    """Expected scores should sum to 1.0."""
    a = expected_score(1600, 1400)
    b = expected_score(1400, 1600)
    assert a + b == pytest.approx(1.0)


def test_update_ratings_winner_gains():
    """Winning team should always gain ELO points."""
    home_new, away_new = update_ratings(1500, 1500, 2, 1, "Friendly")
    assert home_new > 1500  # home team won, should gain
    assert away_new < 1500  # away team lost, should lose


def test_update_ratings_zero_sum():
    """Points gained by winner should equal points lost by loser."""
    home_before, away_before = 1500.0, 1500.0
    home_after, away_after = update_ratings(
        home_before, away_before, 1, 0, "FIFA World Cup"
    )
    gained = home_after - home_before
    lost = away_before - away_after
    assert gained == pytest.approx(lost, abs=0.1)


def test_update_ratings_draw_neutral():
    """Equal rated teams drawing on neutral ground should not change ratings."""
    home_new, away_new = update_ratings(1500, 1500, 1, 1, "Friendly", neutral=True)
    assert home_new == pytest.approx(1500, abs=0.1)
    assert away_new == pytest.approx(1500, abs=0.1)


def test_update_ratings_draw_home_advantage():
    """
    Equal rated teams drawing at home venue — home team loses slight ELO
    because home advantage made them the favourite, so a draw underperforms
    their expected score.
    """
    home_new, away_new = update_ratings(1500, 1500, 1, 1, "Friendly", neutral=False)
    assert home_new < 1500   # home team expected to win, draw is a disappointment
    assert away_new > 1500   # away team exceeded expectations


def test_goal_difference_multiplier_increases_k():
    """A 3-0 win should move ratings more than a 1-0 win."""
    big_win_home, _ = update_ratings(1500, 1500, 3, 0, "FIFA World Cup", neutral=True)
    small_win_home, _ = update_ratings(1500, 1500, 1, 0, "FIFA World Cup", neutral=True)
    assert big_win_home > small_win_home


def test_goal_difference_multiplier_values():
    """Test multiplier values match eloratings.net specification exactly."""
    from src.data.elo import goal_difference_multiplier
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == 1.75
    assert goal_difference_multiplier(4) == pytest.approx(1.875)
    assert goal_difference_multiplier(0) == 1.0  # draws


def test_world_cup_k_higher_than_friendly():
    """World Cup matches should move ratings more than friendlies."""
    # Same result, different tournament
    wc_home, _ = update_ratings(1500, 1500, 1, 0, "FIFA World Cup")
    fr_home, _ = update_ratings(1500, 1500, 1, 0, "Friendly")
    assert wc_home > fr_home  # World Cup win gives more ELO


def test_upset_gives_more_points():
    """Upset win (lower rated team wins) should give more points than expected win."""
    # Lower rated team wins — big gain
    upset_winner_new, _ = update_ratings(1300, 1700, 1, 0, "FIFA World Cup")
    # Higher rated team wins — small gain
    expected_winner_new, _ = update_ratings(1700, 1300, 1, 0, "FIFA World Cup")
    assert (upset_winner_new - 1300) > (expected_winner_new - 1700)


# ── Integration test for calculate_elo ───────────────────────────────────────

@pytest.fixture
def minimal_matches():
    """Three matches between two teams — enough to test ELO progression."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "home_team": ["Brazil", "Argentina", "Brazil"],
        "away_team": ["Argentina", "Brazil", "Argentina"],
        "home_score": [2, 0, 1],
        "away_score": [0, 1, 1],
        "tournament": ["Friendly", "Friendly", "Copa América"],
        "neutral": [False, False, True],
        "is_world_cup": [0, 0, 0],
        "tournament_weight": [0.8, 0.8, 1.3],
    })


def test_calculate_elo_returns_correct_columns(minimal_matches):
    """Output should contain all original columns plus ELO columns."""
    result = calculate_elo(minimal_matches)
    for col in ["home_elo_before", "away_elo_before", "elo_diff",
                "home_elo_after", "away_elo_after"]:
        assert col in result.columns, f"Missing column: {col}"


def test_calculate_elo_initial_rating(minimal_matches):
    """First match for any team should use INITIAL_RATING."""
    result = calculate_elo(minimal_matches)
    first_match = result.iloc[0]
    assert first_match["home_elo_before"] == INITIAL_RATING
    assert first_match["away_elo_before"] == INITIAL_RATING


def test_calculate_elo_no_lookahead(minimal_matches):
    """
    ELO before should never equal ELO after for the same row.
    This confirms we're storing pre-match ratings, not post-match.
    """
    result = calculate_elo(minimal_matches)
    # After first match, Brazil's ELO should have changed
    assert result.iloc[0]["home_elo_before"] != result.iloc[0]["home_elo_after"]


def test_calculate_elo_average_stays_constant(minimal_matches):
    """
    Average ELO across all teams should stay near INITIAL_RATING.
    ELO is zero-sum — points gained equal points lost.
    """
    result = calculate_elo(minimal_matches)
    # Get final ratings for each team
    final_brazil = result[result["home_team"] == "Brazil"].iloc[-1]["home_elo_after"]
    final_argentina = result[result["away_team"] == "Argentina"].iloc[-1]["away_elo_after"]
    avg = (final_brazil + final_argentina) / 2
    assert avg == pytest.approx(INITIAL_RATING, abs=1.0)