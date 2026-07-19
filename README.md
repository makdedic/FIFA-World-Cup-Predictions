# FIFA-World-Cup-Predictions

Predicts win/draw/loss probabilities for international football matches,
built on 150+ years of match history.

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
all international matches, not just World Cup ones — World Cup matches alone
(~1,100 of ~50,000) are too few to train on without overfitting; tournament
context (`is_world_cup`, `match_importance`) is passed in as a feature instead.

Two ways to predict:

- `predict_latest(home, away)` — fast path, uses the pre-trained production
  model (`models/outcome_model.joblib`, produced by `python src/model/train.py`)
- `predict_match(home, away, as_of_date)` — "what would we have predicted
  knowing only what was known as of this date" — retrains on the spot using
  only matches strictly before `as_of_date`

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

## Data

This project uses the International Football Results dataset.

Source:
https://github.com/martj42/international_results

License:
CC0-1.0 (Public Domain)
