"""
predict.py
Predicts win/draw/loss probabilities for a hypothetical match between two
countries, using only data known as of a given date.

"As of" is a hard constraint, not just an evaluation label: features and
training data are both restricted to matches strictly before `as_of_date`,
so a prediction for 2015-06-01 can never be influenced by anything that
happened afterwards, no matter what the full dataset contains.

The hypothetical matchup is featurised by appending it as a placeholder row
and running it through the real build_features() (src/features/engineering.py)
— the same function used to build the training set — rather than a separate
lookup implementation. Training and prediction features must be computed by
identical code, or they silently drift apart.

Two ways to predict, because a future UI (Streamlit) needs to stay responsive
while a user browses many matchups:
  - predict_latest():  fast path. Loads the pre-trained "all data up to the
                        last pipeline run" model from models/outcome_model.joblib
                        (src/model/train.py) — no retraining per click.
  - predict_match():   general path, for an arbitrary historical as_of_date —
                        "what would we have predicted knowing only what was
                        known back then." Retrains, so it's a few seconds.
    train_as_of() / predict_with_model() are the two halves of predict_match()
    split apart — train_as_of() once per as_of_date (cache it), then
    predict_with_model() cheaply per matchup against that cached model.

Run with:
    python src/model/predict.py France Brazil 2026-07-01
    python src/model/predict.py France Brazil --latest
"""

import re
import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import xgboost as xgb
from loguru import logger
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))  # src/model/ — for the sibling `train` module
from train import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    add_match_importance,
    augment_neutral_matches,
    make_xy,
    train_xgb,
)

sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ — for sibling `data`/`features` packages
from data.elo import INITIAL_RATING, get_k_factor
from features.engineering import build_features

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "worldcup.duckdb"
MODELS_DIR = ROOT / "models"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_matches(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the ELO-enriched match table (src/data/pipeline.py's `matches` table)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM matches").df()
    con.close()
    return df


def get_all_teams(matches: pd.DataFrame | None = None, db_path: Path = DB_PATH) -> list[str]:
    """
    Every standardised team name in the dataset (see TEAM_NAME_MAP in
    src/data/clean.py), sorted alphabetically. For populating a dropdown —
    a free-text team field silently mispredicts on a typo (an unrecognised
    name just gets treated as a brand-new team with no history) rather than
    erroring, so a dropdown of real names is the fix, not input validation.
    """
    if matches is None:
        matches = load_matches(db_path)
    return sorted(set(matches["home_team"]) | set(matches["away_team"]))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _elo_as_of(history: pd.DataFrame, team: str) -> float:
    """A team's ELO from its most recent match in `history`, or INITIAL_RATING if unseen."""
    team_matches = history[(history["home_team"] == team) | (history["away_team"] == team)]
    if team_matches.empty:
        return INITIAL_RATING
    last = team_matches.sort_values("date").iloc[-1]
    return last["home_elo_after"] if last["home_team"] == team else last["away_elo_after"]


def _is_world_cup(tournament: str) -> int:
    """Mirrors src/data/clean.py's is_world_cup flag exactly."""
    return int(bool(re.search(r"FIFA World Cup$", tournament, re.IGNORECASE)))


# ── Training (the expensive, cacheable-per-date step) ───────────────────────────

def train_as_of(
    as_of_date: str,
    matches: pd.DataFrame | None = None,
    db_path: Path = DB_PATH,
) -> tuple[XGBClassifier, pd.DataFrame]:
    """
    Trains a model on matches strictly before `as_of_date` and returns it
    alongside that history slice (needed by predict_with_model to featurise
    hypothetical matchups). Cache this per as_of_date in a UI — it's the
    expensive half of a prediction; browsing different matchups at the same
    date shouldn't pay for it again.
    """
    as_of_date = pd.Timestamp(as_of_date)
    if matches is None:
        matches = load_matches(db_path)

    history = matches[matches["date"] < as_of_date]
    if history.empty:
        raise ValueError(f"No matches before {as_of_date.date()} — nothing to train on")

    train_df = add_match_importance(build_features(history))
    train_df = augment_neutral_matches(train_df)

    X_train, y_train = make_xy(train_df)
    model = train_xgb(X_train, y_train)
    return model, history


# ── Prediction (the cheap, per-matchup step) ────────────────────────────────────

def _raw_proba(
    model: XGBClassifier,
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: pd.Timestamp,
    neutral: bool,
    tournament: str,
) -> tuple:
    """
    [away_win, draw, home_win] for exactly this home/away ordering — no
    symmetry correction — plus that same ordering's per-class SHAP feature
    contributions (contribs[class_idx] gives one row per FEATURE_COLUMNS
    entry, in the same 0=away/1=draw/2=home class order as proba). Not part
    of the public API; see predict_with_model, which is what actually
    enforces neutral-venue symmetry and picks which class's contributions
    to surface.
    """
    home_elo = _elo_as_of(history, home_team)
    away_elo = _elo_as_of(history, away_team)

    # neutral/is_world_cup are supplied here (not patched in afterwards) so that
    # concat never introduces NaN into those columns — a bool/int column with even
    # one NaN silently upcasts to dtype=object across the WHOLE column, including
    # history's rows, which XGBoost then rejects at training time.
    hypothetical = pd.DataFrame([{
        "date": as_of_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": 0,   # placeholder — never used as a feature for this row itself
        "away_score": 0,   # placeholder
        "tournament": tournament,
        "neutral": neutral,
        "is_world_cup": _is_world_cup(tournament),
    }])
    featurised = build_features(pd.concat([history, hypothetical], ignore_index=True))

    predict_row = featurised[
        (featurised["date"] == as_of_date)
        & (featurised["home_team"] == home_team)
        & (featurised["away_team"] == away_team)
    ].iloc[-1].copy()
    predict_row["elo_diff"] = home_elo - away_elo
    predict_row["match_importance"] = get_k_factor(tournament)

    # predict_row is a single-row Series pulled from a mixed-dtype DataFrame, so it's
    # boxed as dtype=object — cast to float or XGBoost rejects it as non-numeric.
    X_pred = predict_row[FEATURE_COLUMNS].to_frame().T.astype(float)
    proba = model.predict_proba(X_pred)[0]

    # Exact TreeSHAP contributions, native to XGBoost — no separate `shap`
    # dependency needed, same algorithm it uses under the hood for tree models.
    # Shape (1, n_classes, n_features+1); last column per class is the bias
    # term, dropped here since it's not a feature contribution.
    raw_contribs = model.get_booster().predict(xgb.DMatrix(X_pred), pred_contribs=True)
    contribs = raw_contribs[0, :, :-1]

    return proba, contribs, home_elo, away_elo


def predict_with_model(
    model: XGBClassifier,
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    is_knockout: bool = False,
) -> dict:
    """
    Win/draw/loss probabilities for home_team vs away_team against an
    already-trained model, using `history` (matches strictly before
    as_of_date — same slice train_as_of trained on) to compute ELO and
    engineered features for the hypothetical matchup.

    On a neutral pitch, which team is arbitrarily labelled "home" in the
    data shouldn't change the odds — but every diff feature flips sign
    under a home/away swap, and XGBoost has no constraint forcing it to
    treat a sign-flipped input as a sign-flipped output. Empirically it
    doesn't: swapping home/away for the same two teams under neutral=True
    moved a team's win probability by 10.7pp on average (up to 36pp) across
    a sample of top-team matchups. So for neutral=True, we predict both
    orderings and average them — exact by construction, not a heuristic.
    Non-neutral matches skip this (real home advantage IS asymmetric).

    train_as_of() also mirrors neutral training rows (train.py's
    augment_neutral_matches) so the model is better calibrated to begin
    with — but that's a training-quality improvement, not a guarantee; the
    averaging here is what actually makes the output exactly symmetric.

    is_knockout=True is a heuristic patch, not a validated model fix — see
    the "Known limitation: draws predicted for knockout matches" section of
    the README. `outcome` in the training data already reflects the score
    AFTER extra time (confirmed against shootouts.csv: e.g. the 2022 final
    is recorded 3-3, the real post-ET scoreline, not 90 minutes), so
    home_win/away_win already fully account for team strength across a full
    120 minutes — nothing in "draw" is unaccounted-for skill. What's left in
    "draw" for a knockout match is purely "still level after 120 minutes,"
    i.e. it goes to a penalty shootout — and shootouts are close to a coin
    flip regardless of team quality (a fundamentally different, much less
    skill-driven contest than 120 minutes of open play). So we split that
    remaining draw probability evenly between home_win and away_win rather
    than reallocating it proportionally, which would double-count team
    strength the model has already captured.

    Also returns feature_contributions: exact TreeSHAP contributions toward
    whichever class ends up predicted (highest final probability), from the
    home_team/away_team ordering as given — not re-averaged across the
    neutral-venue swap the way the probabilities are, so treat it as "what
    drove this one calculation" rather than a fully symmetrized explanation.
    is_knockout's post-hoc draw split isn't feature-driven, so it isn't
    reflected here — the contributions explain the underlying classification,
    not the knockout adjustment applied on top of it.
    """
    as_of_date = pd.Timestamp(as_of_date)
    proba, contribs, home_elo, away_elo = _raw_proba(
        model, history, home_team, away_team, as_of_date, neutral, tournament
    )

    if neutral:
        swapped_proba, _, _, _ = _raw_proba(
            model, history, away_team, home_team, as_of_date, neutral, tournament
        )
        away_win = (proba[0] + swapped_proba[2]) / 2
        draw = (proba[1] + swapped_proba[1]) / 2
        home_win = (proba[2] + swapped_proba[0]) / 2
        proba = (away_win, draw, home_win)

    if is_knockout:
        away_win, draw, home_win = proba
        half_draw = draw / 2
        proba = (away_win + half_draw, 0.0, home_win + half_draw)

    predicted_class = int(max(range(3), key=lambda i: proba[i]))
    feature_contributions = sorted(
        (
            {"feature": FEATURE_LABELS[col], "contribution": float(contribs[predicted_class][i])}
            for i, col in enumerate(FEATURE_COLUMNS)
        ),
        key=lambda d: abs(d["contribution"]),
        reverse=True,
    )

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "as_of_date": str(as_of_date.date()),
        "feature_contributions": feature_contributions,
        "home_elo": round(float(home_elo), 1),
        "away_elo": round(float(away_elo), 1),
        "n_training_matches": len(history),
        "away_win_prob": round(float(proba[0]), 4),
        "draw_prob": round(float(proba[1]), 4),
        "home_win_prob": round(float(proba[2]), 4),
    }
    top = feature_contributions[0]
    logger.info(
        f"{home_team} vs {away_team} as of {result['as_of_date']} "
        f"(ELO {result['home_elo']:.0f} vs {result['away_elo']:.0f}, "
        f"trained on {result['n_training_matches']:,} prior matches): "
        f"home={result['home_win_prob']:.1%}  draw={result['draw_prob']:.1%}  "
        f"away={result['away_win_prob']:.1%}  "
        f"(top factor: {top['feature']}, {top['contribution']:+.3f})"
    )
    return result


# ── Convenience wrappers ─────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    as_of_date: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    is_knockout: bool = False,
    matches: pd.DataFrame | None = None,
    db_path: Path = DB_PATH,
) -> dict:
    """
    One-shot version of train_as_of() + predict_with_model() for a single
    arbitrary historical as_of_date. Retrains from scratch every call — fine
    for a CLI/notebook one-off, too slow to call per UI interaction (use
    train_as_of() once + predict_with_model() per matchup instead).
    """
    model, history = train_as_of(as_of_date, matches=matches, db_path=db_path)
    return predict_with_model(
        model, history, home_team, away_team, as_of_date,
        neutral=neutral, tournament=tournament, is_knockout=is_knockout,
    )


def load_production_model(models_dir: Path = MODELS_DIR) -> XGBClassifier:
    """Loads the model src/model/train.py trained on ALL available data and saved to disk."""
    return joblib.load(models_dir / "outcome_model.joblib")


def predict_latest(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    is_knockout: bool = False,
    matches: pd.DataFrame | None = None,
    db_path: Path = DB_PATH,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    Fast path for the common case ("what are the odds using everything we
    know right now"): loads the pre-trained production model instead of
    retraining. Falls back to training on the spot if no saved model is
    found (run `python src/model/train.py` to produce one).
    """
    if matches is None:
        matches = load_matches(db_path)
    as_of_date = matches["date"].max() + pd.Timedelta(days=1)

    try:
        model = load_production_model(models_dir)
    except FileNotFoundError:
        logger.warning(
            "No saved production model at "
            f"{models_dir / 'outcome_model.joblib'} — training on the spot. "
            "Run `python src/model/train.py` to cache one for next time."
        )
        model, _ = train_as_of(as_of_date, matches=matches)

    return predict_with_model(
        model, matches, home_team, away_team, as_of_date,
        neutral=neutral, tournament=tournament, is_knockout=is_knockout,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument(
        "as_of_date", nargs="?",
        help="YYYY-MM-DD — only matches before this date are used. "
             "Omit with --latest to use the pre-trained production model.",
    )
    parser.add_argument("--latest", action="store_true", help="Use the fast pre-trained production model")
    parser.add_argument("--tournament", default="FIFA World Cup")
    parser.add_argument("--not-neutral", action="store_true", help="Match is at home_team's home ground")
    parser.add_argument(
        "--knockout", action="store_true",
        help="Match can't end in a draw (goes to extra time + penalties) — "
             "splits the draw probability evenly into home/away instead",
    )
    args = parser.parse_args()

    if args.latest:
        predict_latest(
            args.home_team, args.away_team,
            neutral=not args.not_neutral, tournament=args.tournament,
            is_knockout=args.knockout,
        )
    else:
        if not args.as_of_date:
            parser.error("as_of_date is required unless --latest is set")
        predict_match(
            args.home_team, args.away_team, args.as_of_date,
            neutral=not args.not_neutral, tournament=args.tournament,
            is_knockout=args.knockout,
        )
