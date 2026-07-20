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

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # src/ — for the `model` package
logger.remove()  # quiet pipeline/model INFO logs in the Streamlit console

from data.elo import K_FACTORS  # noqa: E402
from model.predict import (  # noqa: E402
    DB_PATH,
    get_all_teams,
    load_matches,
    load_production_model,
    predict_with_model,
    train_as_of,
)

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")

# Fixed categorical colors — home/away are competing identities (blue/orange),
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


# ── Guard: pipeline must have been run at least once ────────────────────────

if not DB_PATH.exists():
    st.error(
        f"Database not found at `{DB_PATH}`. Run `python src/data/pipeline.py` "
        "from the project root first."
    )
    st.stop()

matches = cached_matches()
teams = get_all_teams(matches)
latest_date = matches["date"].max()

st.title("⚽ World Cup Match Predictor")
st.caption(
    f"XGBoost model trained on {len(matches):,} international matches "
    f"(1872–{latest_date.year}). For fun — not betting advice."
)


# ── Inputs ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    home_default = teams.index("Argentina") if "Argentina" in teams else 0
    home_team = st.selectbox("Home team", teams, index=home_default)
with col2:
    away_candidates = [t for t in teams if t != home_team]
    away_default_team = "Brazil" if "Brazil" in away_candidates else away_candidates[0]
    away_team = st.selectbox("Away team", away_candidates, index=away_candidates.index(away_default_team))

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

with st.expander("Match context"):
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

predict_clicked = st.button("Predict", type="primary")


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
    labels = bars.mark_text(align="left", dx=6).encode(
        text=alt.Text("Probability:Q", format=".1%"), color=alt.value("#0b0b0b")
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
