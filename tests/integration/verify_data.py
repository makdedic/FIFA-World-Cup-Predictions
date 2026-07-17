# tests/integration/verify_data.py
import pytest
import duckdb
from pathlib import Path

DB_PATH = Path("data/worldcup.duckdb")


@pytest.fixture(scope="module")
def con():
    """Connect to DuckDB — skips all tests if database doesn't exist."""
    if not DB_PATH.exists():
        pytest.skip("Database not found — run pipeline.py first")
    connection = duckdb.connect(str(DB_PATH))
    yield connection
    connection.close()


def test_row_counts(con):
    """Tables should have minimum expected row counts."""
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] >= 45_000
    assert con.execute("SELECT COUNT(*) FROM goalscorers").fetchone()[0] >= 40_000
    assert con.execute("SELECT COUNT(*) FROM shootouts").fetchone()[0] >= 500


def test_date_range(con):
    """Data should span from 1872 to at least 2026."""
    min_d, max_d = con.execute(
        "SELECT MIN(date), MAX(date) FROM matches"
    ).fetchone()
    assert str(min_d) <= "1873-01-01"
    assert str(max_d) >= "2026-01-01", "2026 World Cup data missing"


def test_elo_average_near_1500(con):
    """Average ELO should be ~1500 — confirms zero-sum property."""
    avg = con.execute(
        "SELECT AVG(current_elo) FROM current_rankings"
    ).fetchone()[0]
    assert 1400 < avg < 1600, f"Average ELO is {avg:.0f}, expected ~1500"


def test_elo_top_teams_recognisable(con):
    """At least 3 well-known football nations should be in the top 10."""
    top10 = con.execute(
        "SELECT team FROM current_rankings LIMIT 10"
    ).df()["team"].tolist()
    known_strong = ["Brazil", "Argentina", "France", "Germany", "Spain", "England"]
    strong_in_top10 = [t for t in known_strong if t in top10]
    assert len(strong_in_top10) >= 3, f"Only {strong_in_top10} in top 10"


def test_elo_direction_after_upset(con):
    """After an away upset win, away ELO should increase and home ELO decrease."""
    upset = con.execute("""
        SELECT home_elo_before, home_elo_after,
               away_elo_before, away_elo_after
        FROM world_cup_matches
        WHERE outcome = 0 AND elo_diff > 100
        ORDER BY elo_diff DESC
        LIMIT 1
    """).df().iloc[0]
    assert upset["away_elo_after"] > upset["away_elo_before"]
    assert upset["home_elo_after"] < upset["home_elo_before"]


def test_no_null_elo(con):
    """No nulls in ELO columns — every match should have ratings."""
    nulls = con.execute("""
        SELECT COUNT(*) FROM matches
        WHERE home_elo_before IS NULL OR away_elo_before IS NULL
    """).fetchone()[0]
    assert nulls == 0, f"Found {nulls} matches with null ELO"


def test_world_cup_view_has_2026(con):
    """world_cup_matches view should include 2026 tournament."""
    count = con.execute("""
        SELECT COUNT(*) FROM world_cup_matches
        WHERE date >= '2026-01-01'
    """).fetchone()[0]
    assert count > 0, "No 2026 World Cup matches found in view"


def test_head_to_head_view(con):
    """Head-to-head view should return data for known fixture."""
    count = con.execute("""
        SELECT COUNT(*) FROM head_to_head
        WHERE (team_a = 'England' AND team_b = 'Germany')
           OR (team_a = 'Germany' AND team_b = 'England')
    """).fetchone()[0]
    assert count > 0, "England vs Germany H2H not found"