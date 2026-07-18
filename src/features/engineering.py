"""
engineering.py
Builds per-team and head-to-head features from the ELO-enriched match table.

All features are computed strictly from matches BEFORE the one being featurised —
same no-lookahead discipline as src/data/elo.py. A team's first-ever appearance
therefore has NaN form/streak/H2H features; that's a real "no history" signal,
not a bug, and should be handled at modelling time (impute or let a tree-based
model split on missingness).
"""

import numpy as np
import pandas as pd
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

FORM_WINDOWS = [5, 10]  # matches, for rolling form averages


# ── Long format ───────────────────────────────────────────────────────────────

def _to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape one-row-per-match into two-rows-per-match (one per team's
    perspective). Makes rolling/streak/H2H features a groupby("team")
    away rather than a pair of near-duplicate home/away implementations.

    Expects `matches` to already have a `match_id` column (added by
    build_features before calling this) so features can be merged back.
    """
    home = pd.DataFrame({
        "match_id":      matches["match_id"],
        "date":          matches["date"],
        "team":          matches["home_team"],
        "opponent":      matches["away_team"],
        "is_home":       True,
        "goals_for":     matches["home_score"],
        "goals_against": matches["away_score"],
    })
    away = pd.DataFrame({
        "match_id":      matches["match_id"],
        "date":          matches["date"],
        "team":          matches["away_team"],
        "opponent":      matches["home_team"],
        "is_home":       False,
        "goals_for":     matches["away_score"],
        "goals_against": matches["home_score"],
    })

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(int)
    long_df["unbeaten"] = (long_df["goals_for"] >= long_df["goals_against"]).astype(int)
    long_df["points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"],
         long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )

    return long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


# ── Per-team features ─────────────────────────────────────────────────────────

def _add_rest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    """Days since the team's previous match. NaN on a team's first appearance."""
    long_df["days_since_last_match"] = (
        long_df.groupby("team")["date"].diff().dt.days
    )
    return long_df


def _add_rolling_form(long_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Rolling averages of points/goals/win-rate over the last N matches,
    computed on shift(1) so the match being featurised is never included
    in its own form figures.
    """
    grouped = long_df.groupby("team")
    for window in windows:
        for col, source in [
            ("form_points_avg",       "points"),
            ("form_goals_for_avg",    "goals_for"),
            ("form_goals_against_avg","goals_against"),
            ("form_win_rate",         "win"),
        ]:
            long_df[f"{col}_{window}"] = grouped[source].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return long_df


def _streak_before(flags: pd.Series) -> pd.Series:
    """
    Length of the consecutive-True streak a team carries INTO each row.
    Operates on an already-shift(1)'d boolean series, so the streak at
    row i reflects results strictly before match i.
    """
    shifted = flags.shift(1, fill_value=False).astype(bool)
    reset_groups = (~shifted).cumsum()
    return shifted.groupby(reset_groups).cumsum()


def _add_streaks(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df["win_streak"] = long_df.groupby("team")["win"].transform(
        lambda s: _streak_before(s.astype(bool))
    )
    long_df["unbeaten_streak"] = long_df.groupby("team")["unbeaten"].transform(
        lambda s: _streak_before(s.astype(bool))
    )
    return long_df


def _add_head_to_head(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prior record against this specific opponent (not general form).
    h2h_matches_played counts only past meetings; h2h_win_rate is the
    team's win rate in those meetings.
    """
    h2h = long_df.groupby(["team", "opponent"])
    long_df["h2h_matches_played"] = h2h.cumcount()
    long_df["h2h_win_rate"] = h2h["win"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return long_df


# ── Merge back to wide format ─────────────────────────────────────────────────

_FEATURE_COLS = (
    ["days_since_last_match", "win_streak", "unbeaten_streak",
     "h2h_matches_played", "h2h_win_rate"]
    + [f"form_{stat}_{w}" for w in FORM_WINDOWS
       for stat in ["points_avg", "goals_for_avg", "goals_against_avg", "win_rate"]]
)


def _merge_side(matches: pd.DataFrame, long_df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    prefix = "home_" if is_home else "away_"
    side = long_df.loc[long_df["is_home"] == is_home, ["match_id", *_FEATURE_COLS]]
    side = side.rename(columns={col: f"{prefix}{col}" for col in _FEATURE_COLS})
    return matches.merge(side, on="match_id", how="left")


def _add_diff_features(matches: pd.DataFrame) -> pd.DataFrame:
    for col in _FEATURE_COLS:
        matches[f"{col}_diff"] = matches[f"home_{col}"] - matches[f"away_{col}"]
    return matches


# ── Public API ─────────────────────────────────────────────────────────────────

def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set for match outcome prediction on top of the
    ELO-enriched match table (output of src.data.elo.calculate_elo).

    Adds, per side (home_/away_) and as home-minus-away diffs:
      - days_since_last_match   — fatigue/rest proxy
      - form_{points,goals_for,goals_against,win_rate}_avg_{5,10} — recent form
      - win_streak / unbeaten_streak — momentum going into the match
      - h2h_matches_played / h2h_win_rate — record vs this specific opponent

    Every feature is computed from strictly prior matches (see module docstring).
    Rows are returned in the same order as the input.
    """
    logger.info(f"Building features for {len(matches):,} matches...")

    matches = matches.sort_values("date").reset_index(drop=True)
    matches["match_id"] = matches.index
    long_df = _to_long_format(matches)
    long_df = _add_rest_days(long_df)
    long_df = _add_rolling_form(long_df, FORM_WINDOWS)
    long_df = _add_streaks(long_df)
    long_df = _add_head_to_head(long_df)

    matches = _merge_side(matches, long_df, is_home=True)
    matches = _merge_side(matches, long_df, is_home=False)
    matches = _add_diff_features(matches)
    matches = matches.drop(columns=["match_id"])

    logger.info(
        f"Feature engineering complete — {len(_FEATURE_COLS) * 3} feature columns added "
        f"(home/away/diff for {len(_FEATURE_COLS)} base features)"
    )
    return matches
