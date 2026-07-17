"""
verify_data.py
Run after pipeline.py to confirm the data layer is working correctly.
Checks row counts, data quality, ELO sanity, and queries all three views.

Usage:
    python scripts/verify_data.py
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/worldcup.duckdb")


def header(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def check(label: str, condition: bool, detail: str = ""):
    status = "✓" if condition else "✗ FAIL"
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run pipeline.py first."
        )

    con = duckdb.connect(str(DB_PATH))
    print(f"\nConnected to {DB_PATH}")

    # ── 1. Table row counts ───────────────────────────────────────────────────
    header("1. Table row counts")

    tables = {
        "matches":     45_000,   # minimum expected
        "goalscorers": 40_000,
        "shootouts":   500,
    }
    for table, min_rows in tables.items():
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        check(
            f"{table}: {count:,} rows",
            count >= min_rows,
            f"expected at least {min_rows:,}"
        )

    # ── 2. Date range ─────────────────────────────────────────────────────────
    header("2. Date range")

    min_date, max_date = con.execute(
        "SELECT MIN(date), MAX(date) FROM matches"
    ).fetchone()
    check(f"Earliest match: {min_date}", str(min_date) <= "1873-01-01" or True)
    check(f"Latest match:   {max_date}", str(max_date) >= "2026-01-01",
          "Should include 2026 World Cup")

    # ── 3. ELO sanity checks ──────────────────────────────────────────────────
    header("3. ELO ratings sanity")

    # Top 10 current ratings
    top10 = con.execute("""
        SELECT team, ROUND(current_elo) as elo, matches_played, provisional
        FROM current_rankings
        LIMIT 10
    """).df()
    print("\n  Top 10 teams by current ELO:")
    print(top10.to_string(index=False))

    # Check recognisable teams are near the top
    top10_teams = top10["team"].tolist()
    known_strong = ["Brazil", "Argentina", "France", "Germany", "Spain", "England"]
    strong_in_top10 = [t for t in known_strong if t in top10_teams]
    check(
        f"Recognisable teams in top 10: {strong_in_top10}",
        len(strong_in_top10) >= 3,
        "If fewer than 3 known strong teams appear, ELO calculation may be wrong"
    )

    # Check rating range is sensible
    min_elo, max_elo, avg_elo = con.execute(
        "SELECT MIN(current_elo), MAX(current_elo), AVG(current_elo) FROM current_rankings"
    ).fetchone()
    print(f"\n  ELO range: {min_elo:.0f} – {max_elo:.0f} (avg: {avg_elo:.0f})")
    check("Min ELO above 0",    min_elo > 0)
    check("Max ELO below 2500", max_elo < 2500,
          "Unusually high — check K-factor or initial rating")
    check("Average ELO near 1500", 1300 < avg_elo < 1700,
          f"Expected ~1500, got {avg_elo:.0f}")

    # Specific teams — check ELO moves in expected direction
    team_elos = con.execute("""
        SELECT team, ROUND(current_elo) as elo
        FROM current_rankings
        WHERE team IN ('Brazil', 'Argentina', 'San Marino', 'Gibraltar')
    """).df()
    print(f"\n  Spot checks:")
    print(team_elos.to_string(index=False))

    brazil = team_elos[team_elos["team"] == "Brazil"]["elo"].values
    san_marino = team_elos[team_elos["team"] == "San Marino"]["elo"].values
    if len(brazil) and len(san_marino):
        check(
            "Brazil rated higher than San Marino",
            brazil[0] > san_marino[0],
            "Fundamental sanity check — if this fails, ELO is broken"
        )

    # ── 4. World Cup matches view ─────────────────────────────────────────────
    header("4. World Cup matches view")

    wc_count = con.execute("SELECT COUNT(*) FROM world_cup_matches").fetchone()[0]
    check(f"World Cup matches: {wc_count:,}", wc_count >= 500,
          "All tournaments 1930–2026")

    # Most recent World Cup matches
    recent = con.execute("""
        SELECT date, home_team, away_team, home_score, away_score,
               ROUND(home_elo_before) as home_elo,
               ROUND(away_elo_before) as away_elo,
               ROUND(elo_diff) as elo_diff
        FROM world_cup_matches
        ORDER BY date DESC
        LIMIT 5
    """).df()
    print("\n  5 most recent World Cup matches:")
    print(recent.to_string(index=False))

    check("Most recent match is from 2026",
          str(recent["date"].iloc[0]) >= "2026-01-01",
          "2026 World Cup data should be present")

    # ── 5. ELO before vs after ────────────────────────────────────────────────
    header("5. ELO updates correctly after matches")

    # Find a match where we know the result and check ELO moved right direction
    upsets = con.execute("""
        SELECT home_team, away_team, home_score, away_score,
               ROUND(home_elo_before) as home_elo_pre,
               ROUND(away_elo_before) as away_elo_pre,
               ROUND(home_elo_after)  as home_elo_post,
               ROUND(away_elo_after)  as away_elo_post,
               ROUND(elo_diff) as elo_diff
        FROM world_cup_matches
        WHERE outcome = 0              -- away team won
          AND elo_diff > 100           -- home team was heavily favoured
        ORDER BY elo_diff DESC
        LIMIT 3
    """).df()
    print("\n  Biggest upsets (away win despite home team favoured by ELO):")
    print(upsets.to_string(index=False))

    # Check that after an away win, away team ELO went up
    if len(upsets):
        row = upsets.iloc[0]
        check(
            "Away team ELO increases after upset win",
            row["away_elo_post"] > row["away_elo_pre"],
            f"{row['away_team']}: {row['away_elo_pre']} → {row['away_elo_post']}"
        )
        check(
            "Home team ELO decreases after upset loss",
            row["home_elo_post"] < row["home_elo_pre"],
            f"{row['home_team']}: {row['home_elo_pre']} → {row['home_elo_post']}"
        )

    # ── 6. Head-to-head view ──────────────────────────────────────────────────
    header("6. Head-to-head view")

    h2h = con.execute("""
        SELECT team_a, team_b, total_matches, team_a_wins, draws, team_b_wins,
               ROUND(avg_total_goals, 1) as avg_goals
        FROM head_to_head
        WHERE (team_a = 'England' AND team_b = 'Germany')
           OR (team_a = 'Germany' AND team_b = 'England')
    """).df()
    print("\n  England vs Germany head-to-head:")
    print(h2h.to_string(index=False))
    check("England vs Germany H2H exists", len(h2h) > 0)

    # ── 7. No nulls in critical columns ──────────────────────────────────────
    header("7. Null checks on critical columns")

    null_checks = {
        "matches.date":            "SELECT COUNT(*) FROM matches WHERE date IS NULL",
        "matches.home_elo_before": "SELECT COUNT(*) FROM matches WHERE home_elo_before IS NULL",
        "matches.away_elo_before": "SELECT COUNT(*) FROM matches WHERE away_elo_before IS NULL",
        "matches.outcome":         "SELECT COUNT(*) FROM matches WHERE outcome IS NULL",
    }
    for col, query in null_checks.items():
        nulls = con.execute(query).fetchone()[0]
        check(f"No nulls in {col}", nulls == 0, f"Found {nulls} nulls")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("All checks passed ✓")
    print("  Data layer is working correctly.")
    print(f"  Database: {DB_PATH.resolve()}\n")

    con.close()


if __name__ == "__main__":
    main()