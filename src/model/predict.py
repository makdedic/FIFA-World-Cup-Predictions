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
    python src/model/predict.py 2026-07-01 France Brazil
    python src/model/predict.py --latest France Brazil
"""

import re
import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))  # src/model/ — for the sibling `train` module
from train import FEATURE_COLUMNS, make_xy, train_xgb

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

    train_df = build_features(history)
    train_df["match_importance"] = train_df["tournament"].apply(get_k_factor)

    X_train, y_train = make_xy(train_df)
    model = train_xgb(X_train, y_train)
    return model, history


# ── Prediction (the cheap, per-matchup step) ────────────────────────────────────

def predict_with_model(
    model: XGBClassifier,
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> dict:
    """
    Win/draw/loss probabilities for home_team vs away_team against an
    already-trained model, using `history` (matches strictly before
    as_of_date — same slice train_as_of trained on) to compute ELO and
    engineered features for the hypothetical matchup.
    """
    as_of_date = pd.Timestamp(as_of_date)
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

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "as_of_date": str(as_of_date.date()),
        "home_elo": round(float(home_elo), 1),
        "away_elo": round(float(away_elo), 1),
        "n_training_matches": len(history),
        "away_win_prob": round(float(proba[0]), 4),
        "draw_prob": round(float(proba[1]), 4),
        "home_win_prob": round(float(proba[2]), 4),
    }
    logger.info(
        f"{home_team} vs {away_team} as of {result['as_of_date']} "
        f"(ELO {result['home_elo']:.0f} vs {result['away_elo']:.0f}, "
        f"trained on {result['n_training_matches']:,} prior matches): "
        f"home={result['home_win_prob']:.1%}  draw={result['draw_prob']:.1%}  "
        f"away={result['away_win_prob']:.1%}"
    )
    return result


# ── Convenience wrappers ─────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    as_of_date: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
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
        neutral=neutral, tournament=tournament,
    )


def load_production_model(models_dir: Path = MODELS_DIR) -> XGBClassifier:
    """Loads the model src/model/train.py trained on ALL available data and saved to disk."""
    return joblib.load(models_dir / "outcome_model.joblib")


def predict_latest(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
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
        neutral=neutral, tournament=tournament,
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
    args = parser.parse_args()

    if args.latest:
        predict_latest(
            args.home_team, args.away_team,
            neutral=not args.not_neutral, tournament=args.tournament,
        )
    else:
        if not args.as_of_date:
            parser.error("as_of_date is required unless --latest is set")
        predict_match(
            args.home_team, args.away_team, args.as_of_date,
            neutral=not args.not_neutral, tournament=args.tournament,
        )
