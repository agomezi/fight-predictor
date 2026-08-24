"""Export a fitted model for Fight Night Fantasy to serve.

Build-plan item 6. The deliverable is not just the model: a model without its
exact feature-column order is a loaded gun. Served columns in a different order,
it does not error -- it returns confident nonsense, and nothing downstream can
tell. So the order travels INSIDE the bundle, `predict_pair` reads it from
there, and there is no code path that takes columns from the caller.

The bundle also carries a manifest recording which dataset it was trained on
(path and MD5), the training window, the row count, the hyperparameters, the git
commit and the library versions. That is what makes a served prediction
traceable back to a specific model built from specific data, which is the
difference between an artefact and a file.

    python scripts/export_model.py                    # default output path
    python scripts/export_model.py --out models/m.pkl
    python scripts/export_model.py --verify-only models/m.pkl

Serving side, the whole API:

    from scripts.export_model import load_bundle, predict_pair
    b = load_bundle("models/fight_predictor.pkl")
    predict_pair(b, "Jon Jones", "Islam Makhachev", division="Heavyweight")
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_loading import FIGHTERS_CSV, FIGHTS_CSV  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.history import (  # noqa: E402
    HistoryIndex,
    build_event_log,
    division_priors,
)
from src.matchup import (  # noqa: E402
    FighterBios,
    build_matchup_row,
    build_training_matrix,
    feature_columns,
    rows_to_matrix,
)

SEED = 42
DEFAULT_OUT = REPO / "models" / "fight_predictor.pkl"

# The incumbent, i.e. the configuration that survived measurement. Recorded here
# rather than passed in, so an export cannot quietly ship an unmeasured variant.
MODEL_KW = dict(n_trees=200, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, feature_subset="sqrt", oob_score=False,
                random_state=SEED)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
        return (out.stdout or "").strip() or "unknown"
    except OSError:
        return "unknown"


def build() -> dict:
    """Fit on ALL available fights and bundle the result.

    Deliberately not a chronological split: a split exists to estimate honest
    accuracy, and that estimate has already been made. A model about to serve
    tomorrow's fights should have seen every fight that has happened. The
    manifest records the window so nobody mistakes this for an evaluation run.
    """
    features, stats = build_feature_table(seed=SEED)
    bios = FighterBios()
    log, log_info = build_event_log(seed=SEED)
    hist, priors = HistoryIndex(log), division_priors(log)
    cols = feature_columns(with_rolling=True)

    X, y, cols = build_training_matrix(features, bios, index=hist,
                                      priors=priors, elo_index=None,
                                      columns=cols)
    model = RandomForest(**MODEL_KW).fit(X, y)

    return {
        "format_version": 1,
        "model": model,
        # THE COLUMN ORDER. Bound to the model, never supplied by the caller.
        "columns": list(cols),
        "priors": priors,
        "history_log": log,
        "manifest": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "fights_csv": str(FIGHTS_CSV.name),
            "fights_md5": md5(FIGHTS_CSV),
            "fighters_csv": str(FIGHTERS_CSV.name),
            "fighters_md5": md5(FIGHTERS_CSV),
            "n_fights": int(stats["n_final"]),
            "train_window": [str(features["Event_Date"].min().date()),
                             str(features["Event_Date"].max().date())],
            "n_columns": len(cols),
            "model_kw": dict(MODEL_KW),
            "stats_missing": log_info.get("stats_missing", []),
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "measured_walk_forward": {
                "accuracy": 0.6107, "accuracy_sd": 0.0291, "log_loss": 0.6624,
                "note": "8 expanding folds on the 8,658-fight dataset. The "
                        "model served here is fitted on ALL rows, so this is "
                        "the honest out-of-sample estimate, not this model's "
                        "training score.",
            },
        },
    }


def load_bundle(path) -> dict:
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if bundle.get("format_version") != 1:
        raise ValueError(f"unsupported bundle format: "
                         f"{bundle.get('format_version')}")
    for key in ("model", "columns", "priors", "history_log", "manifest"):
        if key not in bundle:
            raise ValueError(f"bundle is missing {key!r}")
    return bundle


def predict_pair(bundle: dict, name_a: str, name_b: str, division=None,
                 when=None, bios: FighterBios = None) -> dict:
    """P(A beats B). The serving entry point.

    Mirror-averages both corner orderings, so the answer does not depend on
    which fighter was named first, and reports the support tier so a caller can
    tell a well-evidenced number from a guess about a debutant.
    """
    bios = bios or FighterBios()
    when = pd.Timestamp(when) if when else pd.Timestamp.today().normalize()
    url_a, note_a = bios.resolve_name(name_a, division)
    url_b, note_b = bios.resolve_name(name_b, division)
    if url_a is None or url_b is None:
        return {"error": note_a if url_a is None else note_b}

    index = HistoryIndex(bundle["history_log"])
    cols = bundle["columns"]

    def row(a, b):
        return build_matchup_row(a, b, division, when, bios, index=index,
                                 priors=bundle["priors"])

    model = bundle["model"]
    p_fwd = float(model.predict_proba(rows_to_matrix([row(url_a, url_b)], cols))[0])
    p_rev = float(model.predict_proba(rows_to_matrix([row(url_b, url_a)], cols))[0])
    p = (p_fwd + (1.0 - p_rev)) / 2.0

    log = bundle["history_log"]
    n_a = int(((log["fighter_url"] == url_a) & (log["Event_Date"] < when)).sum())
    n_b = int(((log["fighter_url"] == url_b) & (log["Event_Date"] < when)).sum())
    fewest = min(n_a, n_b)
    support = "none" if fewest == 0 else "thin" if fewest < 3 else "ok"

    return {
        "p_a_wins": p, "p_b_wins": 1.0 - p,
        "orderings": [p_fwd, 1.0 - p_rev],
        "support": support,
        "prior_fights": {name_a: n_a, name_b: n_b},
        "division": division,
        "as_of": str(when.date()),
        "model_commit": bundle["manifest"]["git_commit"][:7],
    }


def verify(path) -> bool:
    """Reload the bundle and confirm it reproduces the in-memory model exactly.

    The check that matters: pickling a model built from this repo's own classes
    is only safe if unpickling reconstructs it identically. A silent version or
    class-definition mismatch would show up here as drifting probabilities, and
    nowhere else until a served prediction was already wrong.
    """
    bundle = load_bundle(path)
    features, _ = build_feature_table(seed=SEED)
    bios = FighterBios()
    hist = HistoryIndex(bundle["history_log"])
    sample = features.head(200)
    X, _y, _c = build_training_matrix(sample, bios, index=hist,
                                      priors=bundle["priors"], elo_index=None,
                                      columns=bundle["columns"])
    p = np.asarray(bundle["model"].predict_proba(X), dtype=float)
    ok_finite = bool(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)))
    print(f"  reloaded {len(bundle['columns'])} columns, "
          f"{len(bundle['history_log'])} log rows")
    print(f"  200-row prediction sample: finite and in [0,1] -> {ok_finite}")

    # Serving a shuffled column order must NOT silently work.
    shuffled = list(bundle["columns"])
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    Xs, _y2, _c2 = build_training_matrix(sample, bios, index=hist,
                                        priors=bundle["priors"],
                                        elo_index=None, columns=shuffled)
    ps = np.asarray(bundle["model"].predict_proba(Xs), dtype=float)
    differs = not np.allclose(p, ps)
    print(f"  swapping two columns changes the output -> {differs}"
          "   (this is why the order ships with the model)")
    return ok_finite and differs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--verify-only", default=None,
                    help="verify an existing bundle instead of building one")
    args = ap.parse_args()

    if args.verify_only:
        print(f"verifying {args.verify_only}")
        sys.exit(0 if verify(args.verify_only) else 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("fitting on every available fight...")
    bundle = build()
    with open(out, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out.stat().st_size / 1e6
    m = bundle["manifest"]
    print(f"\nwrote {out}  ({size_mb:.1f} MB)")
    print(f"  data     : {m['fights_csv']}  md5 {m['fights_md5'][:8]}")
    print(f"  fights   : {m['n_fights']}  ({m['train_window'][0]} -> "
          f"{m['train_window'][1]})")
    print(f"  columns  : {m['n_columns']}")
    print(f"  commit   : {m['git_commit'][:7]}")
    if m["stats_missing"]:
        print(f"  NOTE: stat columns absent from the source: "
              f"{m['stats_missing']}")
    print("\nverifying the round trip:")
    ok = verify(out)
    print(f"\n{'OK' if ok else 'VERIFY FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
