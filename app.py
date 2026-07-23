"""
app.py
Streamlit web app for the World Cup match predictor — pick two countries,
a date, and match context, get win/draw/loss probabilities.

Reuses src/model/predict.py's public API directly (train_as_of,
predict_with_model, predict_latest's building blocks) rather than
reimplementing any prediction logic — the app is a UI layer on top of the
same functions the CLI and notebook use.

Run with:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from loguru import logger

PROJECT_ROOT = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src/ — for the `model` package
logger.remove()  # quiet pipeline/model INFO logs in the Streamlit console

from data.elo import K_FACTORS  # noqa: E402
from model.predict import (  # noqa: E402
    DB_PATH,
    MODELS_DIR,
    get_all_teams,
    get_current_team_stats,
    get_head_to_head,
    load_matches,
    load_production_model,
    predict_with_model,
    train_as_of,
)

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")

# Fixed categorical colours — home/away are competing identities (blue/orange),
# draw is a neutral non-competing state (gray), not a third hue to distinguish.
OUTCOME_COLORS = {"Home win": "#2a78d6", "Draw": "#898781", "Away win": "#eb6834"}


# ── Cached data / model loading ─────────────────────────────────────────────
# Streamlit reruns the whole script on every widget interaction, so anything
# expensive (loading matches, training a model) must be cached or the app
# would rebuild the model on every dropdown change.

@st.cache_data
def cached_matches():
    return load_matches()


@st.cache_resource
def cached_production_model():
    try:
        return load_production_model()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Training a model as of this date...")
def cached_model_as_of(as_of_date_str):
    return train_as_of(as_of_date_str, matches=cached_matches())


@st.cache_data(show_spinner="Calculating probabilities...")
def cached_prediction(_model, _history, home_team, away_team, as_of_date, neutral, tournament, is_knockout):
    return predict_with_model(
        _model, _history, home_team, away_team, as_of_date,
        neutral=neutral, tournament=tournament, is_knockout=is_knockout,
    )


# ── Bootstrap: build the database/model on first boot if missing ────────────
# A fresh deploy (e.g. Streamlit Community Cloud) clones the repo with no
# data/ or models/ — both are gitignored, reproduced by these scripts rather
# than committed. Self-heal here instead of just erroring, since a deployed
# visitor has no way to run a separate script themselves. sys.executable
# guarantees the SAME Python/venv the app itself is running under.

def _run_setup_script(relative_path: str, label: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative_path)],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        st.error(f"{label} failed:\n```\n{result.stderr[-2000:]}\n```")
        st.stop()


if not DB_PATH.exists():
    with st.spinner("First-time setup: building the database (~30s, one-time)..."):
        _run_setup_script("src/data/pipeline.py", "Database setup")

if not (MODELS_DIR / "outcome_model.joblib").exists():
    with st.spinner("First-time setup: training the production model (~10s, one-time)..."):
        _run_setup_script("src/model/train.py", "Model training")

matches = cached_matches()
teams = get_all_teams(matches)
latest_date = matches["date"].max()

st.title("⚽ World Cup Match Predictor")
st.write(
    "Predicts the outcome of any international football match — pick two "
    "teams, choose a date, and see win/draw/loss probabilities from a model "
    "trained on 150+ years of match history, plus an explanation of what's "
    "actually driving the prediction."
)


st.markdown("**How to use this**")
st.markdown(
    """
1. Pick a **home** and **away** team.
2. Choose a prediction basis — **"Right now"** uses everything known today;
   **"As of a specific date"** retrains using only matches strictly before
   that date, so you can ask "what would we have predicted before this
   actually happened?"
3. Set the tournament, whether it's a neutral venue, and whether it's a
   knockout tie (can't end in a draw) under **Match context**.
4. Click **Predict** for win/draw/loss probabilities and a breakdown of what
   drove the prediction.
    """
)


st.markdown("**📊 Explore the data**")
st.caption(
    "Current ELO and streaks for every team, ranked by ELO. This is "
    "'right now' data — it ignores any 'as of a specific date' cutoff "
    "set above, so it always reflects everything in the dataset."
)
team_stats = get_current_team_stats(matches)
search = st.text_input("Filter by team name", key="explore_search")
if search:
    team_stats = team_stats[team_stats["team"].str.contains(search, case=False)]

st.dataframe(
    team_stats.rename(columns={
        "team": "Team",
        "current_elo": "ELO",
        "matches_played": "Matches played",
        "provisional": "Provisional",
        "win_streak": "Win streak",
        "unbeaten_streak": "Unbeaten streak",
    }),
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={"ELO": st.column_config.NumberColumn(format="%.0f")},
)
st.caption(f"Showing {len(team_stats):,} of {len(teams):,} teams, from {len(matches):,} matches total.")


# ── Inputs ───────────────────────────────────────────────────────────────────


# Both selectboxes use the SAME full `teams` list (not filtered based on each
# other's current value) and a stable `key` — that's what lets Streamlit
# persist each selection independently across reruns. Dynamically filtering
# one dropdown's options based on the other's value (the previous approach)
# made Streamlit treat the options list as "changed" on every rerun and reset
# to the default instead of remembering what the user had picked.
col1, col2 = st.columns(2)
with col1:
    home_default = teams.index("Argentina") if "Argentina" in teams else 0
    home_team = st.selectbox("Home team", teams, index=home_default, key="home_team")
with col2:
    away_default = teams.index("Brazil") if "Brazil" in teams else 0
    away_team = st.selectbox("Away team", teams, index=away_default, key="away_team")

if home_team == away_team:
    st.warning("Pick two different teams.")

h2h = get_head_to_head(matches, home_team, away_team)
if h2h["total_matches"] > 0:
    st.caption(
        f"**Head-to-head** ({h2h['total_matches']} previous meeting"
        f"{'s' if h2h['total_matches'] != 1 else ''}): "
        f"{home_team} {h2h['team_a_wins']} · Draws {h2h['draws']} · {away_team} {h2h['team_b_wins']}"
    )
    st.markdown(f"**Last {len(h2h['recent_meetings'])} meetings**")
    recent_df = pd.DataFrame(h2h["recent_meetings"]).rename(columns={
        "date": "Date",
        "home_team": "Home",
        "away_team": "Away",
        "home_score": "Home score",
        "away_score": "Away score",
        "tournament": "Tournament",
    })
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
else:
    st.caption("**Head-to-head**: no previous meetings in this dataset.")

mode = st.radio(
    "Prediction basis",
    ["Right now (all available data)", "As of a specific date"],
    horizontal=True,
)

as_of_specific = None
if mode == "As of a specific date":
    as_of_specific = st.date_input(
        "As of date — only matches strictly before this date are used",
        value=latest_date.date(),
    )
    st.caption("Retrains a fresh model for this date — takes a few seconds.")

st.markdown("**Match context**")
# Free text risked a typo silently falling through to get_k_factor()'s
# default K=30 — same failure mode a team free-text field had, same fix.
tournament = st.selectbox("Tournament", list(K_FACTORS.keys()), index=0)
neutral = st.checkbox("Neutral venue", value=True)
is_knockout = st.checkbox(
    "Knockout tie (can't end in a draw)",
    value=False,
    help="Splits the draw probability evenly into home/away instead of "
         "predicting an outcome that can't actually happen — see README.",
)

predict_clicked = st.button("Predict", type="primary", disabled=(home_team == away_team))


# ── Prediction ───────────────────────────────────────────────────────────────

if predict_clicked:
    if mode == "Right now (all available data)":
        as_of_date = latest_date + pd.Timedelta(days=1)
        history = matches
        model = cached_production_model()
        if model is None:
            st.info(
                "No cached production model found — training on the spot "
                "(run `python src/model/train.py` to cache one for next time)."
            )
            model, history = cached_model_as_of(str(as_of_date))
    else:
        as_of_date = pd.Timestamp(as_of_specific)
        try:
            model, history = cached_model_as_of(str(as_of_date))
        except ValueError as e:
            st.error(str(e))
            st.stop()

    result = cached_prediction(
        model, history, home_team, away_team, str(as_of_date),
        neutral, tournament, is_knockout,
    )

    st.subheader(f"{result['home_team']} vs {result['away_team']}")
    st.caption(
        f"As of {result['as_of_date']}  ·  "
        f"ELO {result['home_team']} {result['home_elo']:.0f} vs "
        f"{result['away_team']} {result['away_elo']:.0f}  ·  "
        f"trained on {result['n_training_matches']:,} prior matches"
    )

    chart_df = pd.DataFrame([
        {"Outcome": f"{result['home_team']} win", "Category": "Home win", "Probability": result["home_win_prob"]},
        {"Outcome": "Draw", "Category": "Draw", "Probability": result["draw_prob"]},
        {"Outcome": f"{result['away_team']} win", "Category": "Away win", "Probability": result["away_win_prob"]},
    ])

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4, size=28)
        .encode(
            # paddingInner spaces the three rows apart (fixes bars/labels crowding
            # each other); paddingOuter keeps the outer two rows off the chart edge.
            y=alt.Y(
                "Outcome:N", sort=None, title=None,
                scale=alt.Scale(paddingInner=0.45, paddingOuter=0.3),
            ),
            # Domain runs to 1.12, not 1.0 — headroom so the "%" label after a
            # near-100% bar has room to render instead of getting clipped at
            # the chart edge. Explicit tick values keep the axis reading 0-100%.
            x=alt.X(
                "Probability:Q",
                axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                scale=alt.Scale(domain=[0, 1.12]),
                title=None,
            ),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values())),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=190)
    )
    # No explicit colour — a hardcoded dark hex was illegible in dark mode.
    # Left unset, it inherits Streamlit's automatic light/dark chart theming.
    labels = bars.mark_text(align="left", dx=6).encode(
        text=alt.Text("Probability:Q", format=".1%")
    )
    st.altair_chart(bars + labels, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{result['home_team']} win", f"{result['home_win_prob']:.1%}")
    m2.metric("Draw", f"{result['draw_prob']:.1%}")
    m3.metric(f"{result['away_team']} win", f"{result['away_win_prob']:.1%}")

    # ── Why this prediction? ─────────────────────────────────────────────────
    outcome_probs = {
        f"{result['home_team']} win": result["home_win_prob"],
        "Draw": result["draw_prob"],
        f"{result['away_team']} win": result["away_win_prob"],
    }
    predicted_label = max(outcome_probs, key=outcome_probs.get)

    st.subheader("Why this prediction?")
    caption = f"Top factors pushing the model toward **{predicted_label}** (exact TreeSHAP contributions)."
    if neutral:
        caption += (
            " Computed from one raw home/away ordering, not re-averaged the way the "
            "probabilities above are — see 'Neutral-venue symmetry' below."
        )
    st.caption(caption)

    contrib_df = pd.DataFrame(result["feature_contributions"][:6])
    contrib_df["Direction"] = contrib_df["contribution"].apply(
        lambda v: "Supports" if v > 0 else "Opposes"
    )
    max_abs = contrib_df["contribution"].abs().max() * 1.25

    contrib_chart = (
        alt.Chart(contrib_df)
        .mark_bar(cornerRadiusEnd=3, size=22)
        .encode(
            y=alt.Y(
                "feature:N", title=None, sort=list(contrib_df["feature"]),
                scale=alt.Scale(paddingInner=0.4, paddingOuter=0.3),
                axis=alt.Axis(labelLimit=200),
            ),
            x=alt.X(
                "contribution:Q", title=None,
                scale=alt.Scale(domain=[-max_abs, max_abs]),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Supports", "Opposes"], range=["#2a78d6", "#e34948"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=230)
    )
    st.altair_chart(contrib_chart, use_container_width=True)


with st.expander("About this model"):
    st.markdown(
        f"""
- **ELO ratings** calculated from scratch (eloratings.net methodology):
  tournament-weighted K-factors, goal-difference multiplier, 100-point home
  advantage.
- **Features**: ELO difference, rolling form (5/10 matches), win/unbeaten
  streaks, days since last match, head-to-head record — all computed
  strictly from matches *before* the one being predicted, so nothing leaks
  a match's own result into its own features.
- **Model**: XGBoost multiclass classifier, evaluated with time-based World
  Cup holdouts (train on everything before a tournament, test on that
  tournament) rather than a random split, which would leak future form/ELO
  into training.
- **Why this prediction**: exact TreeSHAP feature contributions, computed
  natively by XGBoost rather than pulling in a separate `shap` dependency.
- **Neutral-venue symmetry**: which team is labelled "home" on a neutral
  pitch shouldn't change the odds. An earlier version of this model got
  that wrong by up to 36 percentage points — fixed both in training
  (mirrored examples) and at serving time (averaging both orderings, exact
  by construction).
- **Knockout matches** can't actually end in a draw — a level score after
  extra time goes to a penalty shootout. The "Knockout tie" toggle splits
  the draw probability evenly between the two teams instead of predicting
  an outcome that can't happen.
- Model production snapshot last refreshed by `python src/model/train.py`;
  data current through **{latest_date.date()}**.

See the project README for the full write-up, including known limitations.
        """
    )
