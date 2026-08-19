# fight-predictor

A UFC fight outcome predictor built on a decision tree written from scratch —
entropy, information gain, recursive splitting, pruning, and a random forest,
with no ML library doing the modelling work.

The goal was to understand the algorithm rather than to import one, so
`scikit-learn` appears only as an independent implementation to check the
results against.

## The problem, and an honest baseline

Predict the winner of a UFC fight from pre-fight information alone.

The label is symmetrised: for every fight a seeded coin flip decides which
corner becomes "fighter A", and all features are computed as A-minus-B
differences. That destroys the corner-position bias in the raw data, which
means:

- **The majority-class baseline is 0.5026, not ~0.63.** Any accuracy near 0.50
  is worthless.
- Published models on comparable fighter-attribute features land around
  **0.58–0.62**.
- Closing betting lines sit near **0.65** — a useful ceiling reference, not a
  target, since the market also sees camp news, injuries and late replacements.

## Results

Chronological split: train 1994–2023 (6,888 fights), test 2023–2026 (1,512).

| Model | Train | Test |
|---|---|---|
| Baseline (majority class) | — | 0.5026 |
| Decision tree, unpruned | 0.9936 | 0.5403 |
| Decision tree, pruned — depth chosen on validation | 0.5769 | 0.5840 |
| Decision tree, fixed depth 4 | — | 0.5952 |
| Random forest (200 trees) | 0.6109 | 0.5972 |

The unpruned tree is the point of the exercise, not an embarrassment: it
memorises 99.4% of the training set and still lands 4.4 points *below* the
pruned tree on unseen fights. Pruning gives up 42 points of training accuracy
to buy those 4.4 points of real accuracy.

Two pruned trees are listed because they answer different questions. The 0.5840
row is the honest one — its `max_depth` was selected on a chronological
validation split carved out of the training window, never on the test set. The
0.5952 row is a hardcoded depth-4 tree, kept only because it is the reference
the forest was originally compared against.

That the hardcoded depth comes out 1.1 points ahead of the selected one is
itself a finding rather than a bug: the gap sits inside the same confidence
interval discussed below, which is what noise-dominated hyperparameter selection
looks like. Choosing `max_depth` on ~1,400 validation rows, at a signal strength
where the best available split is worth about 0.01 bits, is not a reliable
procedure. Distinguishing the two properly needs the evaluation harness that
does not exist yet — bootstrap intervals and walk-forward folds instead of a
single held-out tail.

A 1,512-fight test set carries roughly a ±2.5 point confidence interval, so the
forest's +0.002 over the depth-4 tree is **not** a real difference. That
comparison is also confounded: the depth-4 tree differs from the forest in both
pruning and ensembling. `scripts/train_forest.py` therefore also reports a tree
using the forest's own pruning settings with every feature visible, which is the
comparison that isolates what bagging plus feature subsampling actually buys.

With only 8 features and ~3 sampled per node, many trees in the ensemble draw a
feature subset that is mostly missingness indicators. Features, not model
choice, are the binding constraint here.

### Out-of-bag scoring reads *pessimistic* here

The forest's OOB accuracy is 0.5585 against a test accuracy of 0.5972 — OOB is
3.9 points **low**, not high.

The usual expectation is the opposite: OOB rows are scattered across the whole
training window, so OOB is a random-split estimate, and random splits normally
flatter you relative to a chronological holdout. That effect is real but is
outweighed by a larger one. The training window includes the early era, where
biometrics are sparse and outcomes are noisier; the test window is 2023–2026,
which is modern, well-documented, and simply easier. The test distribution is
not harder than the training distribution — it is easier, and OOB is measuring
the harder pool.

### Feature importance

Gain-weighted across all trees in the forest:

```
age_diff             0.5952  ####################################
height_diff          0.1680  ##########
reach_diff           0.1667  ##########
stance_same          0.0363  ##
reach_diff_missing   0.0163  #
stance_unknown       0.0079
age_diff_missing     0.0075
height_diff_missing  0.0020
```

Age is worth roughly as much as every other static attribute combined. Reach —
the number commentators reach for first — is worth about the same as height,
and a quarter of what age is worth. Age is computed as of the event date, not
from a scraped birthdate against today, so this is not a leak.

## How it works

1. **Load and normalise** — `src/data_loading.py` parses the string-encoded
   biometrics (`5' 11"`, `72.0"`, `155 lbs.`, `38%`) into numbers.
2. **Resolve the join** — fighter names are not unique, so colliding names are
   resolved to the profile whose listed weight best matches the bout's weight
   class, giving exactly one profile per fighter per fight.
3. **Build symmetric features** — reach, height and age differences plus a
   stance matchup, all relative to the randomised fighter A.
4. **Split chronologically** — on an *event-date* boundary, so no single
   night's card is divided between train and test.
5. **Grow the tree** — best-gain split search, pre-pruning via `max_depth`,
   `min_samples_split` and `min_samples_leaf`.
6. **Grow the forest** — bootstrap sampling per tree, a random feature subset
   per node, majority vote, and out-of-bag scoring from the rows each tree
   never saw.

Split search sorts each column once and sweeps the cut point with a running
prefix sum of positive labels, so candidate thresholds are scored in O(1) each
and the cost per feature is O(n log n) rather than O(n²).

## Validation against scikit-learn

`scripts/compare_sklearn.py` fits this tree and `DecisionTreeClassifier(
criterion="entropy")` on identical data with matched hyperparameters.

On **synthetic data with real signal** (0.39 bits of gain at the root), the two
agree on the root split exactly and on **every single test row**.

On the **UFC data** they agree on 85.5% of rows while landing at 0.5952 vs
0.5959 accuracy — statistically the same result from visibly different trees.
That gap is a signal-strength artifact, not a defect. The best available split
in the entire dataset is worth **0.0115 bits** against a root entropy of
0.99999, so the gain surface is nearly flat and hundreds of candidate splits
sit within a hair of one another. Microscopic differences at the root cascade
into different trees by depth 4 and arrive at the same accuracy.

Where the two differ at the root, this implementation's split has the *higher*
gain, and an exhaustive brute-force scan over all 5,336 candidate cut points
confirms it is the true argmax. Why sklearn's Cython splitter selects a
marginally worse cut here is an open question and was not chased further, since
it does not affect the conclusion.

The from-scratch tree fits a depth-4 tree on 6,888 rows about 3x slower than
sklearn's. The gap widens with depth — sklearn's tree is Cython, this one is
NumPy.

## Data leakage — the main thing this project is about

The source dataset makes leakage very easy, and avoiding it was most of the
work:

- **Fighter career columns** (`Wins`, `Losses`, `SLpM`, `Str_Acc`, `TD_Avg`, …)
  are career-to-date *as scraped in 2026*. Joining a fighter's final 29-0
  record onto a 2012 fight tells the model he never went on to lose. None of
  these columns are used.
- **In-fight statistics** (knockdowns, significant strikes, takedowns, control
  time) describe what happened *inside* the fight being predicted. None are
  used.
- **Random train/test splits** let career-shaped features carry future
  information backwards. The split is chronological everywhere, including the
  validation split used for tuning.

Including any of the above raises accuracy. That increase is the tell, not the
result.

## Known limitations

- **Features are static only** — reach, height, age, stance. No form, record,
  or opponent quality yet. As-of-fight-date rolling aggregates plus an Elo
  rating are the next step, and are where the remaining accuracy is.
- **`Reach` is 44% missing** (1,940 of 4,455 fighters). The pipeline carries
  explicit `*_missing` indicator columns rather than imputing silently.
  Missingness plausibly correlates with era and with obscure fighters, so those
  indicators could be measuring something other than reach — though their
  near-zero gain contribution argues against that.
- **The depth sweep in `train_tree.py` is reported on the test set for
  illustration only.** The pruning hyperparameters themselves are tuned on a
  validation split carved chronologically out of the training window, never
  against the test set.
- **A 1,512-fight test set gives roughly a ±2.5 point confidence interval**, so
  differences smaller than that are not distinguishable from noise. Several
  differences reported above fall into that category and are labelled as such.

## Data

Kaggle's comprehensive UFC dataset, scraped from ufcstats.com. Not committed —
download it and place both CSVs in `data/`:

```
data/ufc_gold_dataset_final.csv    8,551 fights, 1994-2026
data/ufc_fighters_final.csv        4,455 fighter profiles
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python scripts/explore_data.py      # data shape, missingness, join sanity
python scripts/build_features.py    # build and check the feature table
python scripts/test_entropy.py      # entropy / information gain checks
python scripts/test_tree.py         # differential test: fast sweep vs naive reference
python scripts/train_tree.py        # unpruned vs pruned, overfitting made visible
python scripts/train_forest.py      # random forest with out-of-bag scoring
python scripts/compare_sklearn.py   # independent check against scikit-learn
```

## Layout

```
src/
  data_loading.py   CSV loading, unit parsing, fighter/bout join
  features.py       matchup features, chronological split, model matrix
  tree.py           entropy, information gain, split search, tree growth
  forest.py         bootstrap sampling, majority vote, out-of-bag scoring
scripts/            exploration, feature build, tests, training, comparison
```
