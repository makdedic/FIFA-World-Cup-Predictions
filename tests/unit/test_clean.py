# tests/test_clean.py
import pandas as pd
import pytest
from src.data.clean import clean_results, TEAM_NAME_MAP


@pytest.fixture
def sample_results():
    """Minimal results DataFrame for testing."""
    return pd.DataFrame({
        "date": ["2022-11-22", "2022-11-26", "2022-11-30"],
        "home_team": ["Argentina", "United States", "IR Iran"],
        "away_team": ["Saudi Arabia", "England", "USA"],
        "home_score": [1, 0, 2],
        "away_score": [2, 1, 0],
        "tournament": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup"],
        "neutral": [True, True, True],
    })


def test_team_names_standardised(sample_results):
    """United States and IR Iran should be standardised to USA and Iran."""
    result = clean_results(sample_results)
    assert "United States" not in result["home_team"].values
    assert "IR Iran" not in result["home_team"].values
    assert "USA" in result["home_team"].values
    assert "Iran" in result["home_team"].values


def test_outcome_column_correct(sample_results):
    """
    Argentina 1-2 Saudi Arabia → outcome = 0 (away win)
    USA 0-1 England → outcome = 0 (away win)
    Iran 2-0 USA → outcome = 2 (home win)
    """
    result = clean_results(sample_results)
    assert result.iloc[0]["outcome"] == 0  # Argentina lost
    assert result.iloc[1]["outcome"] == 0  # USA lost
    assert result.iloc[2]["outcome"] == 2  # Iran won


def test_is_world_cup_flag(sample_results):
    """All matches are World Cup matches — flag should be 1."""
    result = clean_results(sample_results)
    assert result["is_world_cup"].all()


def test_goal_diff_correct(sample_results):
    """goal_diff = home_score - away_score."""
    result = clean_results(sample_results)
    assert result.iloc[0]["goal_diff"] == -1   # 1 - 2
    assert result.iloc[2]["goal_diff"] == 2    # 2 - 0


def test_dates_parsed_as_datetime(sample_results):
    """Dates should be datetime objects, not strings."""
    result = clean_results(sample_results)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_missing_scores_dropped():
    """Rows with missing scores should be dropped."""
    df = pd.DataFrame({
        "date": ["2022-11-22", "2022-11-23"],
        "home_team": ["Brazil", "France"],
        "away_team": ["Serbia", "Australia"],
        "home_score": [2, None],   # second row has no score
        "away_score": [0, None],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "neutral": [True, True],
    })
    result = clean_results(df)
    assert len(result) == 1  # only the complete row survives