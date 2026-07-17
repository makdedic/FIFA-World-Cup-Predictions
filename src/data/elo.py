"""
elo.py
Calculates ELO ratings for all international football teams
from scratch using the Arpad Elo (1978) formula.

Why calculate our own instead of using eloratings.net?
  - No external dependency or manual download step
  - Full transparency — every rating update is traceable
  - We can tune K-factor weighting to our specific use case
  - Strong interview talking point: we understand the algorithm
    rather than treating ratings as a black box

Algorithm:
  Expected = 1 / (1 + 10^((opponent_elo - team_elo) / 400))
  New ELO  = Old ELO + K × (Actual - Expected)

  where Actual = 1.0 (win), 0.5 (draw), 0.0 (loss)
  and   K      = base_k × tournament_weight
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_RATING = 1500.0    # Starting ELO for all teams
BASE_K = 20.0              # Base sensitivity factor
ELO_SCALE = 400.0          # Controls spread of expected score curve

# K multipliers by tournament importance
# Higher K = match has more impact on ratings
# This is a design choice — World Cup results should matter more than friendlies
K_WEIGHTS = {
    "FIFA World Cup":               2.0,   # K = 40
    "FIFA World Cup qualification": 1.5,   # K = 30
    "UEFA Euro":                    1.5,
    "Copa América":                 1.5,
    "Africa Cup of Nations":        1.5,
    "AFC Asian Cup":                1.5,
    "CONCACAF Gold Cup":            1.5,
    "Friendly":                     0.8,   # K = 16 — friendlies matter less
}


# ── Core ELO functions ────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for team A against team B.
    Returns a probability between 0 and 1.

    If ratings are equal: returns 0.5 (50% expected score)
    If A is 200 points higher: returns ~0.76 (76% expected score)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def get_k_factor(tournament: str) -> float:
    """
    Return the K-factor for a given tournament.
    Higher K = more sensitive to results in this tournament.
    """
    for keyword, multiplier in K_WEIGHTS.items():
        if keyword.lower() in tournament.lower():
            return BASE_K * multiplier
    return BASE_K  # default for unlisted tournaments


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool = False,
) -> tuple[float, float]:
    """
    Calculate new ELO ratings after a single match.

    Args:
        home_rating: ELO rating of home team before match
        away_rating: ELO rating of away team before match
        home_score:  Goals scored by home team
        away_score:  Goals scored by away team
        tournament:  Tournament name (affects K-factor)
        neutral:     If True, no home advantage adjustment

    Returns:
        (new_home_rating, new_away_rating)
    """
    k = get_k_factor(tournament)

    # Home advantage: add 100 points to home team's effective rating
    # unless it's a neutral venue (World Cup matches are always neutral)
    home_advantage = 0 if neutral else 100.0
    adjusted_home = home_rating + home_advantage

    # Expected scores
    exp_home = expected_score(adjusted_home, away_rating)
    exp_away = 1.0 - exp_home

    # Actual scores (1=win, 0.5=draw, 0=loss)
    if home_score > away_score:
        actual_home, actual_away = 1.0, 0.0
    elif home_score == away_score:
        actual_home, actual_away = 0.5, 0.5
    else:
        actual_home, actual_away = 0.0, 1.0

    # Update ratings
    new_home = home_rating + k * (actual_home - exp_home)
    new_away = away_rating + k * (actual_away - exp_away)

    return round(new_home, 2), round(new_away, 2)


# ── Main calculation ──────────────────────────────────────────────────────────

def calculate_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ELO ratings for all teams across all historical matches.

    Processes matches chronologically (oldest first).
    For each match, records the BEFORE rating of both teams —
    this is what we use as a feature, since we need the rating
    each team had going INTO the match, not after it.

    Critical: this avoids lookahead bias. We never use a rating
    that includes information from future matches.

    Returns:
        DataFrame with one row per match containing:
          - All original match columns
          - home_elo_before: home team ELO before this match
          - away_elo_before: away team ELO before this match
          - elo_diff: home_elo_before - away_elo_before
          - home_elo_after: home team ELO after this match
          - away_elo_after: away team ELO after this match
    """
    logger.info(f"Calculating ELO ratings for {len(matches):,} matches...")
    logger.info(f"Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")

    # Current ratings dict — updated after each match
    ratings: dict[str, float] = {}

    # Storage for results
    records = []

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        # Initialise any new teams at INITIAL_RATING
        if home not in ratings:
            ratings[home] = INITIAL_RATING
            logger.debug(f"New team: {home} initialised at {INITIAL_RATING}")
        if away not in ratings:
            ratings[away] = INITIAL_RATING
            logger.debug(f"New team: {away} initialised at {INITIAL_RATING}")

        # Record BEFORE ratings (these become features — no lookahead bias)
        home_before = ratings[home]
        away_before = ratings[away]

        # Calculate new ratings
        home_after, away_after = update_ratings(
            home_rating=home_before,
            away_rating=away_before,
            home_score=int(match["home_score"]),
            away_score=int(match["away_score"]),
            tournament=match["tournament"],
            neutral=bool(match.get("neutral", False)),
        )

        # Update current ratings
        ratings[home] = home_after
        ratings[away] = away_after

        # Store record with both before and after ratings
        records.append({
            **match.to_dict(),
            "home_elo_before": home_before,
            "away_elo_before": away_before,
            "elo_diff":        round(home_before - away_before, 2),
            "home_elo_after":  home_after,
            "away_elo_after":  away_after,
        })

    result_df = pd.DataFrame(records)

    # Summary stats
    all_teams = sorted(ratings.keys())
    final_ratings = pd.Series(ratings)
    logger.info(
        f"ELO calculation complete:\n"
        f"  Teams rated: {len(all_teams)}\n"
        f"  Highest rated: {final_ratings.idxmax()} ({final_ratings.max():.0f})\n"
        f"  Lowest rated:  {final_ratings.idxmin()} ({final_ratings.min():.0f})\n"
        f"  Average rating: {final_ratings.mean():.0f}"
    )

    return result_df


def get_current_ratings(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the most recent ELO rating for every team.
    Useful for displaying current team strengths in the dashboard.
    """
    # Get the last match appearance for each team as home or away
    home_ratings = (
        matches_with_elo[["date", "home_team", "home_elo_after"]]
        .rename(columns={"home_team": "team", "home_elo_after": "elo"})
    )
    away_ratings = (
        matches_with_elo[["date", "away_team", "away_elo_after"]]
        .rename(columns={"away_team": "team", "away_elo_after": "elo"})
    )

    all_ratings = pd.concat([home_ratings, away_ratings])

    # Count matches per team before grouping
    match_counts = all_ratings.groupby("team").size().rename("matches_played")

    # Get most recent rating per team
    current = (
        all_ratings
        .sort_values("date")
        .groupby("team")
        .last()
        .rename(columns={"elo": "current_elo"})
        .join(match_counts)  # join the counts back in
        .reset_index()
        .sort_values("current_elo", ascending=False)
        .drop(columns=["date"])
    )
    
    current["provisional"] = current["matches_played"] < 30
    return current