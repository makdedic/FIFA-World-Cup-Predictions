"""
train.py
Trains a multiclass XGBoost classifier to predict match outcomes
(2=home win, 1=draw, 0=away win) from the engineered features in
src/features/engineering.py.

Trained on ALL international matches (match_features), not just World Cup
matches — World Cup matches (~1,100 rows) are far too few to train 20+
features on without overfitting. Tournament context (is_world_cup, neutral,
match_importance) is passed in as a feature instead, so the model can learn
that World Cup matches behave differently rather than being starved of data.

Evaluated with time-based holdouts — train only on matches strictly before
a World Cup, test on that World Cup — never a random split, which would leak
future form/ELO into training.

Training data mirrors every neutral=True match (see augment_neutral_matches)
so the model isn't just guessing which team the source data happened to call
"home" on a neutral pitch. See predict.py's predict_with_model for the
serving-time symmetry averaging that guarantees this regardless of what the
model learned — this augmentation is a complementary training-quality fix,
not the correctness guarantee itself.

The final model (trained on everything) is saved to models/outcome_model.joblib
— that's the "production" snapshot src/model/predict.py's predict_latest()
loads for instant answers, instead of retraining on the spot. Re-run this
script whenever new match data lands to refresh it. For "as of a specific
historical date" predictions, see predict.py's train_as_of() instead — this
script only ever trains on the full dataset.

Run with:
    python src/model/train.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ — for the sibling `data` package
from data.elo import get_k_factor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "worldcup.duckdb"
MODELS_DIR = ROOT / "models"

# ── Feature set ────────────────────────────────────────────────────────────────
# Home-minus-away diffs only — the problem is symmetric (swap home/away and the
# label flips), so diffs carry the signal without doubling collinear columns.
# elo_diff is already a diff (see src/data/elo.py). None of these use the
# match's own result — see src/features/engineering.py for the no-lookahead
# discipline they're built under.

DIFF_FEATURES = [
    "elo_diff",
    "days_since_last_match_diff",
    "win_streak_diff",
    "unbeaten_streak_diff",
    "h2h_matches_played_diff",
    "h2h_win_rate_diff",
    "form_points_avg_5_diff",
    "form_goals_for_avg_5_diff",
    "form_goals_against_avg_5_diff",
    "form_win_rate_5_diff",
    "form_points_avg_10_diff",
    "form_goals_for_avg_10_diff",
    "form_goals_against_avg_10_diff",
    "form_win_rate_10_diff",
]
CONTEXT_FEATURES = ["neutral", "is_world_cup", "match_importance"]
FEATURE_COLUMNS = DIFF_FEATURES + CONTEXT_FEATURES
TARGET = "outcome"

# Human-readable labels for FEATURE_COLUMNS — for explaining an individual
# prediction (see predict.py's feature_contributions) to someone who isn't
# going to know what "form_win_rate_10_diff" means.
FEATURE_LABELS = {
    "elo_diff": "ELO rating gap",
    "days_since_last_match_diff": "Rest advantage",
    "win_streak_diff": "Win streak",
    "unbeaten_streak_diff": "Unbeaten streak",
    "h2h_matches_played_diff": "Head-to-head experience",
    "h2h_win_rate_diff": "Head-to-head win rate",
    "form_points_avg_5_diff": "Recent form — points (last 5)",
    "form_goals_for_avg_5_diff": "Recent form — goals scored (last 5)",
    "form_goals_against_avg_5_diff": "Recent form — goals conceded (last 5)",
    "form_win_rate_5_diff": "Recent form — win rate (last 5)",
    "form_points_avg_10_diff": "Recent form — points (last 10)",
    "form_goals_for_avg_10_diff": "Recent form — goals scored (last 10)",
    "form_goals_against_avg_10_diff": "Recent form — goals conceded (last 10)",
    "form_win_rate_10_diff": "Recent form — win rate (last 10)",
    "neutral": "Neutral venue",
    "is_world_cup": "World Cup context",
    "match_importance": "Tournament importance",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_match_features(db_path: Path) -> pd.DataFrame:
    """Load the full match_features table (all matches, not just World Cup)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM match_features").df()
    con.close()
    return df


def add_match_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric tournament-tier feature reusing elo.py's K-factor table
    (World Cup=60 ... Friendly=20) instead of one-hot encoding the
    hundreds of raw tournament name strings.
    """
    df = df.copy()
    df["match_importance"] = df["tournament"].apply(get_k_factor)
    return df


def augment_neutral_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Doubles up every neutral=True row with a mirrored copy — every diff
    feature negated, outcome flipped (2<->0, draws unchanged) — so the model
    sees explicit both-direction examples for the one case where "home" vs
    "away" labelling is genuinely arbitrary.

    Without this, the model has no reason to treat a sign-flipped input as a
    sign-flipped output: real neutral matches only ever appear once, with
    whichever team the source data happened to call home_team, so nothing in
    training pushes it toward symmetric behaviour. Measured effect on a
    sample of top-team matchups: an average 10.7pp (up to 36pp) swing in a
    team's predicted win probability purely from swapping which side of a
    neutral fixture it's labelled as.

    This complements, not replaces, the serving-time symmetry averaging in
    predict.py's predict_with_model — that guarantees exact output symmetry
    regardless of what the model learned; this gives it better signal to
    learn from in the first place. Expects match_importance already attached
    (add_match_importance) — mirrored rows just inherit it unchanged, since
    match context doesn't change under the swap. Only ever applied to a
    TRAINING split, never to held-out evaluation data.
    """
    neutral_rows = df[df["neutral"]]
    if neutral_rows.empty:
        return df

    mirrored = neutral_rows.copy()
    mirrored[DIFF_FEATURES] = -mirrored[DIFF_FEATURES]
    mirrored[TARGET] = mirrored[TARGET].map({2: 0, 1: 1, 0: 2})
    mirrored["home_team"] = neutral_rows["away_team"]
    mirrored["away_team"] = neutral_rows["home_team"]

    return pd.concat([df, mirrored], ignore_index=True)


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    X, y for training/prediction. NaNs (a team's first-ever appearance —
    see engineering.py) are left as-is; XGBoost splits on missingness
    natively rather than needing imputation.
    """
    return df[FEATURE_COLUMNS], df[TARGET].astype(int)


# ── Model ──────────────────────────────────────────────────────────────────────

def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def baseline_accuracies(test_df: pd.DataFrame) -> dict:
    """
    Naive baselines the model must beat to be worth using:
      - majority_class: always predict the most common outcome in the test set
      - elo_favorite:   predict the higher-ELO team wins, draw if ELO is tied
    """
    y_test = test_df[TARGET].astype(int)
    majority_acc = y_test.value_counts(normalize=True).max()

    elo_pred = np.select(
        [test_df["elo_diff"] > 0, test_df["elo_diff"] < 0], [2, 0], default=1
    )
    elo_acc = accuracy_score(y_test, elo_pred)

    return {"majority_class_acc": majority_acc, "elo_favorite_acc": elo_acc}


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, proba, labels=[0, 1, 2])

    logger.info(f"[{label}] accuracy={acc:.3f}  log_loss={loss:.3f}  n={len(y_test)}")
    logger.info(
        f"[{label}] classification report:\n"
        + classification_report(
            y_test, preds, labels=[0, 1, 2],
            target_names=["away_win", "draw", "home_win"], zero_division=0,
        )
    )
    return {"accuracy": acc, "log_loss": loss}


def evaluate_on_world_cup(df: pd.DataFrame, wc_year: int) -> dict:
    """
    Time-based holdout: train on everything strictly before `wc_year`,
    test on that year's World Cup matches only. Simulates "if we only knew
    what happened before this tournament, how well would we predict it?"

    `df` must already have match_importance attached. The training split is
    neutral-augmented (see augment_neutral_matches); the test split never is
    — evaluating against synthetic mirrored duplicates would double-count
    neutral test matches and distort the reported accuracy/log-loss.
    """
    train_df = df[df["date"].dt.year < wc_year]
    test_df = df[(df["is_world_cup"] == 1) & (df["date"].dt.year == wc_year)]

    if test_df.empty:
        logger.warning(f"No World Cup {wc_year} matches found — skipping holdout")
        return {}

    X_train, y_train = make_xy(augment_neutral_matches(train_df))
    X_test, y_test = make_xy(test_df)

    model = train_xgb(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, label=f"World Cup {wc_year} holdout")

    baselines = baseline_accuracies(test_df)
    logger.info(
        f"[World Cup {wc_year} holdout] baselines: "
        f"majority_class_acc={baselines['majority_class_acc']:.3f}  "
        f"elo_favorite_acc={baselines['elo_favorite_acc']:.3f}"
    )
    return {**metrics, **baselines}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Loading match_features from {DB_PATH}...")
    df = load_match_features(DB_PATH)
    df = add_match_importance(df)
    logger.info(f"Loaded {len(df):,} matches ({df['is_world_cup'].sum():,} World Cup)")

    evaluate_on_world_cup(df, wc_year=2022)
    evaluate_on_world_cup(df, wc_year=2026)

    # Production model — trained on everything, saved for predict.py's fast path.
    logger.info("Training production model on all available data...")
    X_all, y_all = make_xy(augment_neutral_matches(df))
    final_model = train_xgb(X_all, y_all)

    importances = pd.Series(
        final_model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    logger.info(f"Top 10 features by importance:\n{importances.head(10).to_string()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "outcome_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Saved final model to {model_path}")


if __name__ == "__main__":
    main()
