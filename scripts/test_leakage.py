"""The leakage proof for src/history.py. The most important test in the repo.

Everything else here measures how good the model is. This measures whether the
measurement means anything at all. A leaking feature set does not fail loudly --
it produces better numbers, which is the worst possible failure mode, because the
reward for the bug is indistinguishable from the reward for good work.

Two independent angles, because they catch different mistakes.

ANGLE 1 -- THE FABRICATED FUTURE.
Compute a fighter's features as of some bout. Then invent a fight for that
fighter dated AFTER the bout -- a 30-second blowout win -- and recompute. The
features must come out identical. If a single number moves, features_as_of is
reading past the cutoff (the classic `<=`-for-`<` slip).

ANGLE 2 -- THE SHUFFLED-LABEL CONTROL.
Destroy the relationship between features and outcome by permuting the labels,
then train and evaluate normally. A model built on honest features cannot beat
the base rate on shuffled labels. If it does, the features contain the label --
and this catches leaks angle 1 cannot, because it does not care HOW the
information arrived.

Run from the repo root (venv active):
    python scripts/test_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import accuracy, bootstrap_ci  # noqa: E402
from src.features import FEATURE_NAMES, build_feature_table, to_matrix  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.history import (  # noqa: E402
    AS_OF_FEATURES,
    HistoryIndex,
    build_event_log,
    division_priors,
    features_as_of,
)
from src.matchup import (  # noqa: E402
    ROLLING_DIFF_NAMES,
    FighterBios,
    build_matchup_row,
    rows_to_matrix,
)

failures, skipped = [], []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def skip(name, why):
    print(f"[SKIP] {name}  ({why})")
    skipped.append(name)


def _scalar_equal(x, y) -> bool:
    """Equality that treats NaN-and-NaN as equal.

    Debut and unknown values are NaN, and `np.nan == np.nan` is False, so a
    naive `==` reports a phantom difference (and a `!=`-based test would report
    a phantom PASS). Strings (the support tier) compare by ordinary equality.
    """
    x_nan = isinstance(x, float) and np.isnan(x)
    y_nan = isinstance(y, float) and np.isnan(y)
    if x_nan or y_nan:
        return x_nan and y_nan
    return x == y


def _features_equal(a: dict, b: dict) -> tuple[bool, str]:
    """Compare two feature dicts key by key, NaN-aware. Returns (ok, first_diff)."""
    keys = set(a) | set(b)
    for k in sorted(keys):
        if not _scalar_equal(a.get(k, np.nan), b.get(k, np.nan)):
            return False, f"{k}: {a.get(k)!r} != {b.get(k)!r}"
    return True, ""


# ---------------------------------------------------------------------------
# Written for you: the fabrication helper
# ---------------------------------------------------------------------------
def fabricate_future_fight(log: pd.DataFrame, fighter_url: str,
                           after_date, days: int = 30) -> pd.DataFrame:
    """Return a copy of `log` with one invented fight appended for `fighter_url`.

    The invented fight is deliberately extreme -- a 30-second knockout win with
    lopsided stats -- so that any feature which reads it will move visibly.
    """
    row = {c: np.nan for c in log.columns}
    row.update({
        "fighter_url": fighter_url,
        "opponent_url": "http://x/fabricated-opponent",
        "Fight_URL": "http://f/fabricated",
        "Event_Date": pd.Timestamp(after_date) + pd.Timedelta(days=days),
        "Weight_Class": str(log["Weight_Class"].iloc[0]),
        "won": 1,
        "is_finish": True,
        "Method": "KO/TKO",
        "End_Round": 1,
        "fight_secs": 30.0,
    })
    for c in log.columns:
        if c.startswith("own_"):
            row[c] = 999.0
        elif c.startswith("opp_"):
            row[c] = 0.0
    out = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    return (out.sort_values(["fighter_url", "Event_Date"], kind="mergesort")
               .reset_index(drop=True))


def pick_subject(log: pd.DataFrame):
    """A fighter with several fights, and a cutoff partway through their career."""
    counts = log["fighter_url"].value_counts()
    eligible = counts[counts >= 3]
    if eligible.empty:
        raise RuntimeError("data has no fighter with 3+ fights")
    url = str(eligible.index[0])
    dates = sorted(log.loc[log["fighter_url"] == url, "Event_Date"])
    cut = dates[len(dates) // 2]
    return url, cut, sum(d < cut for d in dates)


log, info = build_event_log(seed=42)
priors = division_priors(log)
subject, as_of, n_prior = pick_subject(log)
subject_wc = str(log.loc[log["fighter_url"] == subject, "Weight_Class"].iloc[0])

print("=" * 78)
print("SETUP")
print("=" * 78)
print(f"log rows        : {info['n_log_rows']} over {info['n_fighters']} fighters")
print(f"subject         : {subject}")
print(f"cutoff date     : {pd.Timestamp(as_of).date()}")
print(f"prior fights    : {n_prior}  (with later fights also on record)")


# ---------------------------------------------------------------------------
# ANGLE 1
# ---------------------------------------------------------------------------
def test_fabricated_future():
    """Inventing a future fight must not change any past feature."""
    base_idx = HistoryIndex(log)
    baseline = features_as_of(base_idx, subject, as_of,
                              weight_class=subject_wc, priors=priors)

    # A boundary bug (`<=`) shows up for a fight one day later and can hide
    # behind a 30-day gap, so test both. Each rebuild uses a FRESH index over
    # the modified log -- reusing base_idx would test nothing.
    for days in (1, 30):
        mod = fabricate_future_fight(log, subject, as_of, days=days)
        mod_idx = HistoryIndex(mod)
        after = features_as_of(mod_idx, subject, as_of,
                               weight_class=subject_wc, priors=priors)
        ok, diff = _features_equal(baseline, after)
        check(f"future fight (+{days}d) leaves past features unchanged", ok, diff)

    # The other direction: the fabricated row MUST be visible from a cutoff
    # after it. Without this, a features_as_of that ignored the log entirely
    # would pass the invariance checks above for the wrong reason.
    mod = fabricate_future_fight(log, subject, as_of, days=30)
    mod_idx = HistoryIndex(mod)
    later = pd.Timestamp(as_of) + pd.Timedelta(days=60)
    n_base = features_as_of(base_idx, subject, later,
                            weight_class=subject_wc, priors=priors)["n_fights"]
    n_mod = features_as_of(mod_idx, subject, later,
                           weight_class=subject_wc, priors=priors)["n_fights"]
    check("the fabricated fight IS counted from a later cutoff",
          n_mod == n_base + 1, f"(base {n_base} -> mod {n_mod})")


# ---------------------------------------------------------------------------
# ANGLE 2
# ---------------------------------------------------------------------------
def _build_rolling_matrix():
    """Static + rolling feature matrix for every fight, plus labels and dates.

    Built once and reused across the shuffle seeds. Elo is left at its initial
    value here (elo_ratings=None): the control is about whether the ROLLING
    RATE features can conjure signal from noise, and a constant column cannot.
    """
    features, _ = build_feature_table(seed=42)
    bios = FighterBios()
    idx = HistoryIndex(log)

    rows = [
        build_matchup_row(r.fighter_A_url, r.fighter_B_url, r.Weight_Class,
                          r.Event_Date, bios, index=idx, priors=priors)
        for r in features.itertuples(index=False)
    ]
    x_static = rows_to_matrix(rows, FEATURE_NAMES)
    x_roll = np.array([[r[n] for n in ROLLING_DIFF_NAMES] for r in rows], dtype=float)
    X = np.hstack([x_static, x_roll])
    y = features["label"].to_numpy(dtype=int)
    order = np.argsort(features["Event_Date"].to_numpy(), kind="mergesort")
    return X[order], y[order], features["Event_Date"].to_numpy()[order]


def test_shuffled_label_control():
    """Honest features must learn nothing once the labels are shuffled."""
    X, y, _dates = _build_rolling_matrix()  # already chronologically ordered
    cut = int(len(y) * 0.82)  # last ~18% is the test tail, matching v1
    inside = 0
    seeds = (1, 2, 3, 4, 5)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Shuffle ONLY y, and only after the features are built. Permuting the
        # whole label array, then splitting, keeps the split honest.
        y_shuf = y[rng.permutation(len(y))]
        y_tr, y_te = y_shuf[:cut], y_shuf[cut:]
        forest = RandomForest(n_trees=60, max_depth=12, min_samples_split=10,
                              min_samples_leaf=5, feature_subset="sqrt",
                              oob_score=False, random_state=seed).fit(X[:cut], y_tr)
        p = np.asarray(forest.predict_proba(X[cut:]), dtype=float)
        base_acc = accuracy(y_te, np.full(len(y_te), float(y_te.mean())))
        _, lo, hi = bootstrap_ci(y_te, p, accuracy, n_boot=1000,
                                 rng=np.random.default_rng(seed))
        got = accuracy(y_te, p)
        # The leak signature is ONE-SIDED: a model that has learned the label
        # scores ABOVE the base rate, i.e. its whole CI sits above it (lo >
        # base_acc). Matching or underperforming the base rate is the healthy
        # result -- with the labels shuffled there is nothing left to learn, so
        # the model can only tie or, from fitting noise, do slightly worse.
        no_leak = base_acc >= lo - 1e-9
        inside += int(no_leak)
        tag = "ok" if no_leak else "BEATS BASE (leak?)"
        print(f"    seed {seed}: acc {got:.3f}  base {base_acc:.3f}  "
              f"CI [{lo:.3f}, {hi:.3f}]  {tag}")
    # One permutation landing high is noise; a real leak clears the base rate
    # across most seeds. Require the model NOT to beat the base rate for the
    # majority of them.
    check("shuffled-label model does not beat the base rate",
          inside >= 4, f"({inside}/{len(seeds)} seeds clear)")


for name, fn in (("fabricated-future invariance", test_fabricated_future),
                 ("shuffled-label control", test_shuffled_label_control)):
    print()
    print("=" * 78)
    print(name.upper())
    print("=" * 78)
    try:
        fn()
    except NotImplementedError:
        skip(name, "TODO(human) -- see the docstring")

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    print("A leak makes the model look BETTER. Do not tune around this.")
    sys.exit(1)
if skipped:
    print(f"{len(skipped)} leakage proof(s) not yet written: {', '.join(skipped)}")
    sys.exit(1)
print("All leakage proofs pass.")
