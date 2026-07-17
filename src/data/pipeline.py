"""
pipeline.py
Prefect orchestration flow for the WorldCup Predictor data layer.

Run with:
    python src/data/pipeline.py

This will:
  1. Download raw data from martj42 GitHub (no API keys needed)
  2. Clean and standardise
  3. Calculate ELO ratings from scratch
  4. Load everything into DuckDB
  5. Create SQL views ready for feature engineering

The entire pipeline runs in ~30 seconds on first run,
~5 seconds on subsequent runs (incremental loading).
"""

from prefect import flow, task, get_run_logger
import pandas as pd
import duckdb
from pathlib import Path

from ingest import ingest_all
from clean import clean_results, clean_goalscorers, clean_shootouts, save_processed
from elo import calculate_elo, get_current_ratings

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "worldcup.duckdb"


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="Ingest raw data", retries=3, retry_delay_seconds=10)
def task_ingest() -> dict[str, pd.DataFrame]:
    logger = get_run_logger()
    logger.info("Starting ingestion from martj42 GitHub...")
    return ingest_all(RAW)


@task(name="Clean and standardise")
def task_clean(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger = get_run_logger()
    logger.info("Cleaning data...")

    results = clean_results(raw["results.csv"])
    goalscorers = clean_goalscorers(raw["goalscorers.csv"])
    shootouts = clean_shootouts(raw["shootouts.csv"])

    save_processed(results,     "matches",     PROCESSED)
    save_processed(goalscorers, "goalscorers", PROCESSED)
    save_processed(shootouts,   "shootouts",   PROCESSED)

    logger.info("Cleaning complete")
    return {
        "matches":     results,
        "goalscorers": goalscorers,
        "shootouts":   shootouts,
    }


@task(name="Calculate ELO ratings")
def task_elo(clean_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Calculating ELO ratings from scratch...")

    matches_with_elo = calculate_elo(clean_data["matches"])
    current_ratings = get_current_ratings(matches_with_elo)

    save_processed(matches_with_elo, "matches_with_elo", PROCESSED)
    save_processed(current_ratings,  "current_elo",      PROCESSED)

    logger.info(
        f"ELO calculation complete — "
        f"top team: {current_ratings.iloc[0]['team']} "
        f"({current_ratings.iloc[0]['current_elo']:.0f})"
    )
    return matches_with_elo


@task(name="Load to DuckDB")
def task_load(
    matches_with_elo: pd.DataFrame,
    clean_data: dict[str, pd.DataFrame],
) -> None:
    logger = get_run_logger()
    logger.info(f"Loading to DuckDB at {DB_PATH}...")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # Load tables
    tables = {
        "matches":          matches_with_elo,
        "goalscorers":      clean_data["goalscorers"],
        "shootouts":        clean_data["shootouts"],
    }
    for name, df in tables.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        logger.info(f"  Loaded {name}: {count:,} rows")

    # ── Create useful views ───────────────────────────────────────────────────

    # View 1: World Cup matches only, with ELO and context
    con.execute("""
        CREATE OR REPLACE VIEW world_cup_matches AS
        SELECT
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            outcome,
            goal_diff,
            total_goals,
            tournament,
            neutral,
            home_elo_before,
            away_elo_before,
            elo_diff,
            home_elo_after,
            away_elo_after
        FROM matches
        WHERE is_world_cup = 1
        ORDER BY date
    """)

    # View 2: Current ELO rankings (most recent rating per team)
    con.execute("""
        CREATE OR REPLACE VIEW current_rankings AS
        WITH home_ratings AS (
            SELECT home_team AS team, date, home_elo_after AS elo
            FROM matches
        ),
        away_ratings AS (
            SELECT away_team AS team, date, away_elo_after AS elo
            FROM matches
        ),
        all_ratings AS (
            SELECT * FROM home_ratings
            UNION ALL
            SELECT * FROM away_ratings
        )
        SELECT
            team,
            LAST(elo ORDER BY date) AS current_elo,
            COUNT(*) AS matches_played
        FROM all_ratings
        GROUP BY team
        ORDER BY current_elo DESC
    """)

    # View 3: Head-to-head record between any two teams
    con.execute("""
        CREATE OR REPLACE VIEW head_to_head AS
        SELECT
            LEAST(home_team, away_team)    AS team_a,
            GREATEST(home_team, away_team) AS team_b,
            COUNT(*) AS total_matches,
            SUM(CASE
                WHEN home_team < away_team AND outcome = 2 THEN 1
                WHEN home_team > away_team AND outcome = 0 THEN 1
                ELSE 0 END) AS team_a_wins,
            SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) AS draws,
            SUM(CASE
                WHEN home_team < away_team AND outcome = 0 THEN 1
                WHEN home_team > away_team AND outcome = 2 THEN 1
                ELSE 0 END) AS team_b_wins,
            AVG(total_goals) AS avg_total_goals
        FROM matches
        GROUP BY team_a, team_b
    """)

    con.close()
    logger.info(
        f"DuckDB loaded successfully.\n"
        f"  Tables: matches, goalscorers, shootouts\n"
        f"  Views:  world_cup_matches, current_rankings, head_to_head\n"
        f"  Database: {DB_PATH}"
    )


# ── Main flow ─────────────────────────────────────────────────────────────────

@flow(name="WorldCup Data Pipeline", log_prints=True)
def run_pipeline():
    """
    Main Prefect flow — orchestrates the full data layer.
    Run this file directly: python src/data/pipeline.py
    """
    raw_data = task_ingest()
    clean_data = task_clean(raw_data)
    matches_with_elo = task_elo(clean_data)
    task_load(matches_with_elo, clean_data)


if __name__ == "__main__":
    run_pipeline()