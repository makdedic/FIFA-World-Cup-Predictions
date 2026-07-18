"""
clean.py
Cleans and standardises raw football data.
Handles team name inconsistencies across 150 years of records,
adds derived columns used in feature engineering downstream.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# ── Team name standardisation ─────────────────────────────────────────────────
# Historical records use different names for the same nation.
# We standardise to modern FIFA naming conventions.

TEAM_NAME_MAP = {
    # Common variations
    "United States":            "USA",
    "IR Iran":                  "Iran",
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    "Türkiye":                  "Turkey",
    "China PR":                 "China",
    "Chinese Taipei":           "Taiwan",
    "Trinidad and Tobago":      "Trinidad & Tobago",
    "Bosnia and Herzegovina":   "Bosnia & Herzegovina",
    "Saint Kitts and Nevis":    "St. Kitts & Nevis",
    "Saint Lucia":              "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",
    # Historical names
    "West Germany":             "Germany",
    "East Germany":             "Germany",   # Note: creates duplicates pre-1990
    "Soviet Union":             "Russia",
    "Yugoslavia":               "Serbia",
    "Czechoslovakia":           "Czech Republic",
    "Zaire":                    "DR Congo",
}


# ── Cleaning functions ────────────────────────────────────────────────────────

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the main results DataFrame.

    Adds:
      - outcome:         2=home win, 1=draw, 0=away win
      - goal_diff:       home_score - away_score
      - is_world_cup:    boolean flag
      - wc_stage:        World Cup stage (Group/R32/R16/QF/SF/Final)
    """
    logger.info(f"Cleaning results: {len(df):,} rows")
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Standardise team names
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # Drop rows where scores are missing
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing scores")

    # Ensure scores are integers
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Validate no negative scores
    assert (df["home_score"] >= 0).all(), "Negative home scores found"
    assert (df["away_score"] >= 0).all(), "Negative away scores found"

    # Derived columns
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"],
         df["home_score"] == df["away_score"]],
        [2, 1],        # 2=home win, 1=draw
        default=0      # 0=away win
    )
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    # World Cup flags
    df["is_world_cup"] = df["tournament"].str.contains(
        "FIFA World Cup$", case=False, na=False
    ).astype(int)

    # World Cup stage classification
    stage_map = {
        "group stage":    "Group",
        "round of 32":    "R32",
        "round of 16":    "R16",
        "quarter-final":  "QF",
        "semi-final":     "SF",
        "third-place":    "3rd",
        "final":          "Final",
    }
    if "stage" in df.columns:
        df["wc_stage"] = df["stage"].str.lower().map(
            lambda s: next(
                (v for k, v in stage_map.items() if isinstance(s, str) and k in s),
                "Group"
            )
        )
    else:
        # martj42 dataset has no stage column — default all WC matches to Group
        # This will be enriched later when we add tournament-specific data
        df["wc_stage"] = "Group"

    # Sort chronologically — important for ELO calculation
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Cleaning complete: {len(df):,} matches "
        f"({df['date'].min().year}–{df['date'].max().year}), "
        f"{df['is_world_cup'].sum():,} World Cup matches"
    )
    return df


def clean_goalscorers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean goalscorers data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["own_goal"] = df["own_goal"].fillna(False).astype(bool)
    df["penalty"] = df["penalty"].fillna(False).astype(bool)
    return df.sort_values("date").reset_index(drop=True)


def clean_shootouts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean penalty shootout data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
    df["winner"] = df["winner"].replace(TEAM_NAME_MAP)
    return df.sort_values("date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str, processed_dir: Path) -> None:
    """Save cleaned DataFrame to parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {name}.parquet ({len(df):,} rows, {path.stat().st_size/1024:.0f}KB)")