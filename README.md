# FIFA-World-Cup-Predictions

Predicts win/draw/loss probabilities for international football matches,
built on 150+ years of match history.

## Setup

Every command in this README assumes the virtual environment is active.
`streamlit`/`python` outside it resolve to a *different* Python (e.g. a
system install or pyenv shim) with none of this project's dependencies —
that's the #1 source of `ModuleNotFoundError` here, not a broken install.

```bash
python3 -m venv .venv
source .venv/bin/activate          # every command below assumes this is active
pip install -r requirements-dev.txt  # includes requirements.txt + test/lint tools
```

## Pipeline

A Prefect flow (`src/data/pipeline.py`) does the heavy lifting, in order:

1. **Ingest** — download raw results from the martj42 GitHub dataset
2. **Clean** — standardise team names, derive `outcome`/`goal_diff`/World Cup flags
3. **ELO** — calculate ELO ratings from scratch, match by match
4. **Feature engineering** — build form/streak/head-to-head features
5. **Load** — write everything to DuckDB (`data/worldcup.duckdb`), with SQL views for querying

Run it with:

```
python src/data/pipeline.py
```

## ELO Ratings

ELO ratings are calculated from scratch using the eloratings.net methodology
(K-factors by tournament, goal difference multiplier, 100-point home advantage).
Absolute values differ slightly from eloratings.net due to 150 years of
historical data including high-scoring early matches — the relative differences
between teams are internally consistent and used as features, not the absolute values.

## Feature Engineering

`src/features/engineering.py` builds per-team and head-to-head features —
recent form (points/goals/win rate over the last 5 and 10 matches), win/unbeaten
streaks, days since last match, and head-to-head record against the specific
opponent. Every feature is computed strictly from matches *before* the one
being featurised, so nothing ever leaks a match's own result into its features.

## Model

`src/model/` trains an XGBoost multiclass classifier on ELO + engineered
features to predict match outcome (home win / draw / away win). Trained on
all international matches, not just World Cup ones as World Cup matches alone
are too few (~1,100 of ~50,000)  to train on without overfitting; tournament
context (`is_world_cup`, `match_importance`) is passed in as a feature instead.

Two ways to predict:

- `predict_latest(home, away)` — fast path, uses the pre-trained production
  model (`models/outcome_model.joblib`, produced by `python src/model/train.py`)
- `predict_match(home, away, as_of_date)` — "what would we have predicted
  knowing only what was known as of this date" — retrains on the spot using
  only matches strictly before `as_of_date`

### Explainability

Every prediction includes `feature_contributions` — exact TreeSHAP values
(XGBoost computes these natively via `pred_contribs=True`, so no separate
`shap` dependency is needed) showing what pushed the model toward whichever
outcome it predicted, ranked by impact with labels
(`FEATURE_LABELS` in `src/model/train.py`). The web app renders this as a
"Why this prediction?" chart. One caveat: for neutral matches this reflects
one raw home/away ordering, not re-averaged the way the reported
probabilities are — see "Neutral-venue symmetry" below for why that
averaging exists in the first place.

### Command line

```
# One-time setup: build the database, then (optionally) cache a production model
python src/data/pipeline.py
python src/model/train.py

# Fast path — uses everything known right now
python src/model/predict.py France Brazil --latest

# "As of" a specific date — retrains using only matches before it
python src/model/predict.py Argentina "Saudi Arabia" 2022-11-20

# Neutral venue is the default; add --not-neutral for a true home fixture,
# --tournament to change context (default "FIFA World Cup")
python src/model/predict.py England Germany 2018-06-01 --not-neutral --tournament Friendly

# Knockout tie — can't end in a draw, so that probability splits into home/away
python src/model/predict.py Argentina "Saudi Arabia" 2022-11-20 --knockout
```

Team names must match the dataset's standardised naming (`TEAM_NAME_MAP` in
`src/data/clean.py`) — e.g. `"USA"` not `"United States"`. Quote names with
spaces, as with `"Saudi Arabia"` above.

Model quality is evaluated with time-based holdouts (train on everything before
a World Cup, test on that World Cup) rather than a random split, which would
leak future form/ELO into training. On the 2022 World Cup holdout, the model
(51.6% accuracy) is roughly tied with a naive "pick the ELO favourite" baseline
(51.6%) and beats a majority-class baseline (43.8%) — a useful sanity check
before trusting it over simpler heuristics.

### Neutral-venue symmetry

On a neutral pitch, which team the data happens to label "home" is arbitrary
and shouldn't change the predicted odds. It did — up to 36 percentage points
on some matchups — because every model feature is a home-minus-away diff, and
nothing forces a tree-based model to treat a sign-flipped input as a
sign-flipped output. Fixed two ways:

- **Training**: `augment_neutral_matches` (`src/model/train.py`) mirrors every
  `neutral=True` training row (diffs negated, outcome flipped, draws
  unchanged) so the model has explicit both-direction examples to learn from.
- **Serving**: `predict_with_model` (`src/model/predict.py`) predicts both
  home/away orderings for neutral matches and averages them — exact by
  construction, independent of what the model actually learned.

### Known limitation: instability late in a knockout run

Predictions can swing sharply from one `as_of_date` to the next when the
cutoff crosses a match either team played — expected in general (ELO/form
legitimately update), but the swing is often larger than the ELO change alone
would suggest. Example: moving the cutoff one day past a real Argentina 2–1
England semi-final changed `elo_diff` by a reasonable +50, but `win_streak_diff`
jumped from 9 to 14 and `unbeaten_streak_diff` from 5 to 14 — and the model's
predicted draw probability jumped from 26% to 50%.

Cause: by the semi-final/final stage, both remaining teams have long unbroken
win streaks by construction (you can't reach the final without one), pushing
streak features into a region that's rare in training — only 4.4% of matches
(2,198 of 49,519) have `|win_streak_diff|` or `|unbeaten_streak_diff| ≥ 10`.
Tree-based models make discontinuous, threshold-based splits, so landing in
that thin, high-variance region can produce an output swing disproportionate
to the actual size of the underlying event. Net effect: predictions are least
stable exactly when a tournament is at its most decisive stage. Not yet
addressed — candidate fixes include capping/log-scaling the streak features
or bucketing them into coarser bins.

### Known limitation: draws predicted for knockout matches (partially addressed)

Knockout matches (Round of 16 onward) can't actually end in a draw — a level
score after extra time goes to a penalty shootout, so there's always a
winner. The model doesn't inherently know this and, by default, predicts a
draw probability for a Round of 16 fixture same as it would for a
group-stage or friendly match.

`home_score`/`away_score` (and therefore `outcome`) already reflect the score
**after extra time**, not just 90 minutes — confirmed by cross-referencing
`shootouts.csv`: the 2022 final is recorded 3-3, the real post-extra-time
scoreline, before Argentina won on penalties. That means `home_win`/`away_win`
already fully account for team strength across a full 120 minutes of play —
nothing is missing there. What's left in `draw`, specifically for a knockout
match, is purely "still level after 120 minutes," i.e. it goes to a penalty
shootout — and shootouts are well-documented as close to a coin flip
regardless of team quality, a fundamentally different contest than 120
minutes of open play.

**Fix (heuristic, not a trained model change)**: `predict_with_model` /
`predict_match` / `predict_latest` accept `is_knockout=True`
(CLI: `--knockout`). When set, the draw probability is split *evenly* into
home/away rather than reallocated proportionally to the existing win/loss
split — proportional would double-count team strength the model has already
captured in reaching that draw estimate.

This requires the caller to say `is_knockout=True` explicitly — there's still
no automatic way to detect knockout stage from the data, since `wc_stage` is
a stub, always `"Group"` (the source dataset has no stage column; see the
comment in `clean_results`). A full fix would still need real per-match stage
data to auto-detect this, plus merging `shootouts.csv` into training so the
model itself learns the knockout-specific
relationship.

## Web App

A Streamlit UI (`app.py`, repo root) on top of the same `src/model/predict.py`
functions the CLI and notebook use — pick two teams, a date, and match
context (neutral venue, knockout tie, tournament), get a win/draw/loss chart
plus a "Why this prediction?" breakdown of the top contributing features.

```bash
python src/data/pipeline.py   # one-time setup, if not already done
python src/model/train.py     # optional — caches a production model for the fast path
streamlit run app.py
```

Opens at `http://localhost:8501`. The "Right now" mode uses the cached
production model (instant); "As of a specific date" retrains on the spot
(a few seconds) — both call straight into `train_as_of`/`predict_with_model`,
so there's no separate prediction logic to keep in sync with the CLI.

### Deployment

`data/worldcup.duckdb` and `models/outcome_model.joblib` are committed —
deliberately, unlike everything else under `data/`/`models/`. A platform
like Streamlit Community Cloud clones the repo fresh on every deploy, wake
from sleep, and cold start; without a committed snapshot, every one of those
would need to rebuild the database and retrain the model from scratch first
(tens of seconds), and there's no way to predict when a visitor will land on
one of those cold starts. `app.py` still self-heals via `_run_setup_script`
if either file is ever missing (e.g. deploying somewhere that doesn't get
the committed copy), but in normal operation that path never runs.

This means the deployed data goes stale until refreshed deliberately —
rerun `python src/data/pipeline.py` and `python src/model/train.py`, then
commit the updated files, whenever you want the live app caught up on
recent results.

## Data

This project uses the International Football Results dataset.

Source:
https://github.com/martj42/international_results

License:
CC0-1.0 (Public Domain)
