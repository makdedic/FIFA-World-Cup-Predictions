"""
elo.py
Calculates ELO ratings for all international football teams
from scratch using the Arpad Elo (1978) formula.

Why calculate our own instead of using eloratings.net?
  - No external dependency or manual download step
  - Full transparency — every rating update is traceable
  - We can tune K-factor weighting to our specific use case

Algorithm:
  Expected = 1 / (1 + 10^((opponent_elo - team_elo) / 400))
  New ELO  = Old ELO + K × (Actual - Expected)

  where Actual = 1.0 (win), 0.5 (draw), 0.0 (loss)
  and   K      = get_k_factor(tournament)

Note: absolute rating values will differ from eloratings.net due to
differences in historical data coverage and early match handling.
The relative differences between teams are consistent and valid
as predictive features.
"""

import pandas as pd
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_RATING = 1500.0    # Starting ELO for all teams
BASE_K = 20.0              # Base sensitivity factor
ELO_SCALE = 400.0          # Controls spread of expected score curve
HOME_ADVANTAGE = 100.0

# K-factors per tournament — matches eloratings.net exactly. Keys are matched
# against this dataset's actual tournament strings (see src/data/clean.py),
# which don't always match eloratings.net's own naming — e.g. this dataset
# says "African Cup of Nations", "Gold Cup", and "Confederations Cup" where
# eloratings.net's write-up says "Africa Cup of Nations", "CONCACAF Gold
# Cup", and "FIFA Confederations Cup".
K_FACTORS = {
    "FIFA World Cup":                     60,
    "UEFA Euro":                          50,
    "Copa América":                       50,
    "African Cup of Nations":             50,
    "AFC Asian Cup":                      50,
    "Gold Cup":                           50,
    "Confederations Cup":                 50,
    "FIFA World Cup qualification":       40,
    "UEFA Euro qualification":            30,
    "AFC Asian Cup qualification":        30,
    "African Cup of Nations qualification": 30,
    "Gold Cup qualification":             30,
    "Friendly":                           20,
}


# ── Core ELO functions ────────────────────────────────────────────────────────

def expected_score(rating_home: float, rating_away: float) -> float:
    """
    Calculate expected score for home team against away team.
    Returns a probability between 0 and 1.

    If ratings are equal: returns 0.5 (50% expected score)
    If A is 200 points higher: returns ~0.76 (76% expected score)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_away - rating_home) / ELO_SCALE))


def get_k_factor(tournament: str) -> float:
    """
    Return K-factor for a given tournament matching eloratings.net methodology.

    Checks the longest keyword first, since e.g. "FIFA World Cup" is a
    substring of "FIFA World Cup qualification" — checking shorter keywords
    first would match every qualifier to its parent tournament's K instead
    of its own.
    """
    for keyword, k in sorted(K_FACTORS.items(), key=lambda item: len(item[0]), reverse=True):
        if keyword.lower() in tournament.lower():
            return float(k)
    return 30.0  # default for unlisted tournaments


def goal_difference_multiplier(goal_diff: int) -> float:
    """
    Goal difference multiplier — matches eloratings.net exactly.

    Win by 1: ×1.0
    Win by 2: ×1.5
    Win by 3: ×1.75
    Win by 4+: ×1.75 + (margin - 3) / 8
    """
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    elif margin == 2:
        return 1.5
    elif margin == 3:
        return 1.75
    else:
        return 1.75 + (margin - 3) / 8


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
    Matches eloratings.net methodology exactly:
      - Tournament-specific K-factors
      - Goal difference multiplier
      - 100-point home advantage on non-neutral venues
    """
    k = get_k_factor(tournament)

    # Home advantage
    home_advantage = 0.0 if neutral else HOME_ADVANTAGE
    adjusted_home = home_rating + home_advantage

    # Expected scores
    exp_home = expected_score(adjusted_home, away_rating)
    exp_away = 1.0 - exp_home

    # Actual scores
    if home_score > away_score:
        actual_home, actual_away = 1.0, 0.0
    elif home_score == away_score:
        actual_home, actual_away = 0.5, 0.5
    else:
        actual_home, actual_away = 0.0, 1.0

    # Goal difference multiplier — applied to K
    gdm = goal_difference_multiplier(home_score - away_score)
    effective_k = k * gdm

    # Update ratings
    new_home = home_rating + effective_k * (actual_home - exp_home)
    new_away = away_rating + effective_k * (actual_away - exp_away)

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