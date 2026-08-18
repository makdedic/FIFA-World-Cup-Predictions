# tests/integration/test_app.py
"""
Boots the actual Streamlit app (app.py) headlessly and checks it renders
without throwing — the class of failure a unit test can't catch, since
app.py's own import graph, caching, and bootstrap logic are never exercised
by testing src/ in isolation. This is what would have caught the
"cannot import name 'get_current_team_stats'" deploy failure before it
reached production.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DB_PATH = Path("data/worldcup.duckdb")
MODEL_PATH = Path("models/outcome_model.joblib")


@pytest.fixture
def app():
    if not DB_PATH.exists() or not MODEL_PATH.exists():
        pytest.skip("Database/model not found — run pipeline.py and train.py first")
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    return at


def test_app_boots_without_error(app):
    assert not app.exception


def test_app_renders_title_and_team_selectors(app):
    assert app.title[0].value == "⚽ World Cup Match Predictor"
    selectbox_keys = {sb.key for sb in app.selectbox}
    assert {"home_team", "away_team"} <= selectbox_keys
