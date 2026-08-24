# fight-predictor

A UFC fight outcome predictor where the decision tree, the random forest and the
gradient booster are all written from scratch — entropy, information gain,
recursive splitting, pruning, bootstrap aggregation, and boosting on the negative
gradient of log loss. `scikit-learn` appears only as an independent
implementation to check the results against.

The interesting part is not the model. It is that **the measuring instrument was
built before the things it measures**, and then used to retire most of my own
ideas — including the two I was most confident about.

---

## Results

Data: 8,658 UFC fights, 1994-03-11 → 2026-08-15. Evaluated with **eight
expanding-window walk-forward folds**, not a single held-out tail.

| feature set / model | accuracy | sd | log loss |
|---|---|---|---|
| coin flip (majority class) | 0.5026 | — | 0.6931 |
| static biometrics only (8 cols) | 0.5728 | 0.0232 | 0.6794 |
| **+ rolling as-of-fight form (22 cols)** | **0.6107** | 0.0291 | **0.6624** |
| gradient boosting, same features | 0.6082 | 0.0241 | 0.6615 |

**0.611 accuracy / 0.662 log loss** is the honest number. For scale:

- **0.50** is the floor. The label is symmetrised — a seeded coin flip decides
  which corner becomes "fighter A" and every feature is an A-minus-B difference
  — so the ~63% corner bias in the raw data is destroyed and there is no free
  accuracy to collect.
- **0.58–0.62** is where published models on comparable features land.
- **~0.66** is closing betting lines. That is the practical ceiling, not a
  target: the market also sees camp news, injuries and late replacements, and a
  flash knockout is not predictable from any feature.

---

## What was measured and retired

Every idea below was implemented, measured with a **paired bootstrap on the same
rows** and across the **fold-to-fold spread**, and kept only if it cleared both.
Almost nothing did.

| idea | result | why it failed |
|---|---|---|
| Live Elo ratings | 0.5987 — retired | 0.84 collinear with `win_rate_diff`, which is the better label predictor of the two |
| Column pruning | 0.6035 — retired | removing collinear columns did not free the per-node feature sample |
| Bout context (title / women's / rounds) | 0.5999 — retired | bout-level, so symmetric across corners: a title-bout flag cannot say who wins |
| Weight, 4 columns | 0.6045 — retired | near-zero on the 78% of fights inside a division |
| `weight_diff` alone | 0.6100 — parity | indistinguishable from the incumbent |
| `cut_burden_diff` alone | **real regression** | the absolute-distance transform discards which fighter is heavier |
| Gradient boosting | 0.6082 — retired | better log loss, worse accuracy, both inside the noise |
| Forest vs a tuned single tree | +0.002 | the forest ties the best tree; bagging buys robustness, not peak accuracy |
| Forest/booster blend | inside noise | beats the booster on log loss, not the forest |
| Platt / isotonic calibration | inside noise | the model was already well calibrated — see below |

**The one thing that worked** was rolling as-of-fight form: +0.038 accuracy over
static features, and it won **8 of 8 folds**. Everything since has been noise.

### Two lessons that cost real work to learn

**A clean number can point the wrong way.** Column pruning had the best
single-tail log loss of any variant and was *worse* than the incumbent on the
fold mean. That failure mode is the entire reason the walk-forward harness
exists.

**Testing features in bundles can bury a good one — but so can reading noise as
signal.** Weight was measured as four columns and read as "does not pay"; split
apart, one column looked like a free win at +0.0003. Against a fold sd of 0.026
that is not a win either. Both readings were errors in opposite directions, and
only re-measuring caught the second one.

### What the model actually keys on

Gain-weighted importance across the forest, on the 22-column incumbent:

```
age_diff              0.1126  ##################
win_rate_diff         0.0799  #############
sig_absorbed_pm_diff  0.0733  ############
sig_landed_pm_diff    0.0614  ##########
td_landed_p15m_diff   0.0593  #########
win_rate_raw_diff     0.0587  #########
```

Age is the single heaviest feature, and it was also the heaviest on the
static-only model where it took 0.595 of the total — roughly as much as every
other biometric combined. Reach, the number commentators reach for first, was
worth about the same as height and a quarter of what age is worth. Age is
computed as of the event date rather than from a scraped birthdate against
today, so it is not a leak.

Worth noting `sig_absorbed_pm_diff` outranking `sig_landed_pm_diff`: how much a
fighter gets hit predicts better than how much they land.

### Out-of-bag scoring reads *pessimistic* here

The forest's OOB accuracy is 0.5585 against a test accuracy of 0.5972 — OOB is
3.9 points **low**, not high.

The usual expectation is the opposite: OOB rows are scattered across the whole
training window, so OOB is a random-split estimate, and random splits normally
flatter you relative to a chronological holdout. That effect is real but is
outweighed by a larger one. The training window opens in 1994, where biometrics
are sparse and outcomes noisier; the test window is modern and well-documented.
The test distribution is not harder than the training distribution — it is
easier, and OOB is scoring the harder pool.

### Validated against scikit-learn

`scripts/compare_sklearn.py` fits this tree and `DecisionTreeClassifier(
criterion="entropy")` on identical data with matched hyperparameters.

On **synthetic data with real signal** (0.39 bits at the root) the two agree on
the root split exactly and on **every test row**.

On the **UFC data** they agree on 85.5% of rows while landing at 0.5952 against
0.5959 accuracy — the same result from visibly different trees. That gap is a
signal-strength artefact, not a defect: the best available split in the entire
dataset is worth **0.0115 bits** against a root entropy of 0.99999, so the gain
surface is nearly flat, hundreds of candidate cuts sit within a hair of the
argmax, and a microscopic difference at the root cascades into a different tree
by depth 4. The verdict in that script is gated on root gain for exactly this
reason — thresholding row agreement at this signal strength measures
tie-breaking, not correctness.

---

## How leakage is prevented

The reason a naive version of this project reports 85% accuracy is data leakage,
and in this sport **a leak looks like success**. Three defences, all enforced by
tests:

**Career averages are excluded.** The source `fighters.csv` carries `Wins`,
`Losses`, `SLpM` and friends, scraped in 2026. Joining those onto a 2012 fight
tells the model the fighter went on to never lose. They have never been in the
feature set.

**Every rolling feature is computed strictly before the bout date.** Not `<=` —
a row sharing the date is either the fight itself or same-card information nobody
had beforehand. `src/history.features_as_of` enforces this per row.

**`scripts/test_leakage.py` proves it two independent ways.** It invents a
30-second blowout win dated *after* a bout and requires the past features to come
back byte-identical; and it shuffles the labels and requires the model to score
no better than the base rate. Both must pass before any number is trusted.

The harness that judges everything lives in `src/evaluate.py`: log loss and Brier
alongside accuracy, bootstrap confidence intervals, a **paired** bootstrap that
resamples once and scores both models on the same rows (13–15× tighter than
comparing two independent intervals), and walk-forward folds.

---

## Known limitations

Stated plainly, because they bound what the number means.

**Five of fourteen rolling features go stale.** `ufcstats.com` — the only source
publishing per-fight statistics — now serves a JavaScript proof-of-work
interstitial, and working around bot detection is out of scope. Weekly refreshes
come from Wikipedia, which carries results but not strike counts. So
`sig_landed_pm`, `sig_absorbed_pm`, `td_landed_p15m`, `sub_att_p15m` and
`ctrl_frac` freeze for any fighter who has fought since the last full-stat
snapshot. The other nine, including every result-derived feature, refresh
normally.

**Cross-division predictions extrapolate.** The model can see weight, and
`cut_burden_diff` responds to the contracted division. But only 198 fights have a
burden gap above 30 lb, mostly heavyweight bouts rather than real weight cuts, so
for something like a lightweight champion at heavyweight the model is answering
outside its training distribution. It should say so; it currently does not.

**Reach is missing on 12% of fights.** 24% of active fighters have no reach
recorded, concentrated in the 1990s. Those 633 cannot be backfilled — the source
is behind the challenge above.

**Half the dataset has a thin corner.** 25.5% of fights involve a UFC debutant
and 55.7% have a side with under three prior fights. Those fighters fall back to
a division prior, so the model is running largely on biometrics for them.
Pre-UFC records would fix this and are the best remaining idea, unbuilt.

**Calibration was measured and not shipped.** Fitted on a chronological
validation fold, Platt scaling moved test log loss by −0.0010 and isotonic made
it worse. The reliability table shows why: the populated probability bins were
already within 0.01–0.04 of observed. Averaging 200 trees calibrates fairly well
on its own, so there was little to correct. A parameter that buys nothing does
not ship.

---

## Repository

```
src/
  tree.py        entropy, information gain, O(n log n) split search, pruning
  forest.py      bootstrap aggregation, OOB scoring, feature importance
  boosting.py    gradient boosting with Newton leaves, monotone constraints
  history.py     per-fighter event log, as-of features, Elo, shrinkage
  features.py    matchup features, chronological split, the leakage boundary
  matchup.py     ONE row builder shared by training and serving
  evaluate.py    metrics, bootstrap + paired bootstrap, walk-forward folds
  calibrate.py   Platt (hand-written) and isotonic
scripts/
  test_*.py      the suites, including the leakage proof
  evaluate_models.py   the ruler
  predict_card.py      two names in, a calibrated probability out
  refresh_data.py      append newly-completed events
  export_model.py      bundle the model WITH its column order
```

`src/matchup.py` exists to prevent training/serving skew: if training and
prediction build a feature row differently, served probabilities go quietly wrong
while every test metric stays clean. One function builds both, and
`scripts/test_matchup.py` rebuilds all 8,658 historical fights through the
serving path and requires zero delta.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/test_leakage.py        # the gate, ~30s
python scripts/evaluate_models.py     # the ruler, ~14 min
python scripts/predict_card.py "Islam Makhachev" "Ilia Topuria" --with-history
```

The datasets are not committed. See `data/PROVENANCE.md` for sources, checksums
and the note on why the original scrape is not reproducible.

## Data

Kaggle's comprehensive UFC dataset (scraped from ufcstats.com), refreshed
forward from Wikipedia's *List of UFC events* under CC BY-SA. A weekly GitHub
Actions job discovers new bouts and opens a **pull request** rather than pushing,
so a human reviews the diff before anything reaches the training data — that
being the one place silent corruption could enter.
