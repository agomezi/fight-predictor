"""Score the model on recent cards, as it would have predicted them on fight week.

    python scripts/score_cards.py                       # cards after 2026-10-03
    python scripts/score_cards.py --since 2026-10-03 --until 2026-11-07
    python scripts/score_cards.py --weight              # add the weight columns

For each card in the window, the incumbent model ("+ rolling form", the one the
walk-forward gate measures) is refit on fights STRICTLY BEFORE that card's date,
then asked about every bout on the card. Nothing from the card itself, or from
any later card, reaches the fit:

  * training rows are sliced by Event_Date < card date;
  * every feature row, including the card's own, is built as of its own date
    through build_matchup_row + HistoryIndex, which test_leakage.py checks;
  * support tiers count only fights before the card.

Each bout is mirror-averaged over both corner orderings, as predict_card does.
Draws and no-contests are dropped by the label cleaner: there is no winner to
score.

Read the output for what it is: a handful of cards. The binomial sd of accuracy
at n ~ 60 is about 6 points, so a card going 9/13 or 5/13 says almost nothing
about the model. The walk-forward gate is the measurement; this is a check that
fight-week predictions behave like it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import brier_score, log_loss, support_tier  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.history import HistoryIndex, build_event_log, division_priors  # noqa: E402
from src.matchup import (  # noqa: E402
    FighterBios,
    build_matchup_row,
    build_training_matrix,
    feature_columns,
    rows_to_matrix,
)

RANDOM_SEED = 42
BASE_RATE_ACC = 0.5019   # always-predict-the-base-rate, walk-forward harness


def prior_counts(features: pd.DataFrame, before: pd.Timestamp) -> dict:
    """Logged fights per fighter strictly before `before`."""
    past = features[features["Event_Date"] < before]
    counts: dict = {}
    for col in ("fighter_A_url", "fighter_B_url"):
        for url, n in past[col].value_counts().items():
            counts[url] = counts.get(url, 0) + int(n)
    return counts


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    return acc, float(log_loss(y, p)), float(brier_score(y, p))


def report(label: str, y: np.ndarray, p: np.ndarray) -> None:
    n = len(y)
    if n == 0:
        return
    acc, ll, br = metrics(y, p)
    sd = math.sqrt(acc * (1 - acc) / n) if 0 < acc < 1 else float("nan")
    print(f"  {label:<20}{n:>4}  {acc:>6.3f} (sd {sd:.3f})  "
          f"{ll:>8.4f}  {br:>7.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2026-10-03",
                    help="score cards strictly after this date (default 2026-10-03)")
    ap.add_argument("--until", default=None,
                    help="score cards on or before this date (default: all)")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--weight", action="store_true",
                    help="add the weight / cut-burden columns, as predict_card does")
    args = ap.parse_args()

    since = pd.Timestamp(args.since)
    until = pd.Timestamp(args.until) if args.until else None

    features, _ = build_feature_table(seed=RANDOM_SEED)
    window = features["Event_Date"] > since
    if until is not None:
        window &= features["Event_Date"] <= until
    cards = sorted(features.loc[window, "Event_Date"].unique())
    if not cards:
        print(f"no labelled bouts after {since.date()}", file=sys.stderr)
        sys.exit(1)

    print("building per-fighter history...", file=sys.stderr, flush=True)
    bios = FighterBios()
    log, _info = build_event_log(seed=RANDOM_SEED)
    index, priors = HistoryIndex(log), division_priors(log)
    cols = feature_columns(with_rolling=True, with_weight=args.weight)

    # Every row is built as of its own date, so one matrix over all fights is
    # leak-free; per card it is sliced to rows strictly before the card.
    X_all, y_all, cols = build_training_matrix(features, bios, index=index,
                                               priors=priors, columns=cols)
    dates = features["Event_Date"].to_numpy()

    out = []
    for card in cards:
        card = pd.Timestamp(card)
        train = dates < np.datetime64(card)
        print(f"{card.date()}: fitting {args.trees} trees on {int(train.sum())} "
              f"fights...", file=sys.stderr, flush=True)
        forest = RandomForest(n_trees=args.trees, max_depth=12,
                              min_samples_split=10, min_samples_leaf=5,
                              feature_subset="sqrt", oob_score=False,
                              random_state=RANDOM_SEED).fit(X_all[train], y_all[train])

        bouts = features[features["Event_Date"] == card]
        fwd = rows_to_matrix(
            [build_matchup_row(r.fighter_A_url, r.fighter_B_url, r.Weight_Class,
                               card, bios, index=index, priors=priors)
             for r in bouts.itertuples(index=False)], cols)
        rev = rows_to_matrix(
            [build_matchup_row(r.fighter_B_url, r.fighter_A_url, r.Weight_Class,
                               card, bios, index=index, priors=priors)
             for r in bouts.itertuples(index=False)], cols)
        p = (forest.predict_proba(fwd) + (1.0 - forest.predict_proba(rev))) / 2.0

        counts = prior_counts(features, card)
        for r, p_a in zip(bouts.itertuples(index=False), p):
            out.append(dict(
                date=card.date(),
                a=bios.name_of(r.fighter_A_url), b=bios.name_of(r.fighter_B_url),
                p_a=float(p_a), y=int(r.label),
                depth_a=counts.get(r.fighter_A_url, 0),
                depth_b=counts.get(r.fighter_B_url, 0),
            ))

    df = pd.DataFrame(out)
    df["tier"] = support_tier(df["depth_a"], df["depth_b"], how="min")
    df["pick"] = np.where(df["p_a"] >= 0.5, df["a"], df["b"])
    df["p_pick"] = np.maximum(df["p_a"], 1.0 - df["p_a"])
    df["winner"] = np.where(df["y"] == 1, df["a"], df["b"])
    df["hit"] = df["pick"] == df["winner"]

    print()
    for card, g in df.groupby("date"):
        print(f"--- {card}  ({int(g['hit'].sum())}/{len(g)})")
        for r in g.itertuples(index=False):
            print(f"  {'HIT ' if r.hit else 'MISS'}  {r.a} vs {r.b}: picked "
                  f"{r.pick} {r.p_pick:.3f}, won {r.winner}  [{r.tier}]")
        print()

    y, p = df["y"].to_numpy(), df["p_a"].to_numpy()
    print("=" * 74)
    print(f"  {'':<20}{'n':>4}  {'accuracy':<18}{'logloss':>8}  {'brier':>7}")
    report("all bouts", y, p)
    for tier in ("ok", "thin", "none"):
        m = (df["tier"] == tier).to_numpy()
        report(f"tier {tier}", y[m], p[m])
    m = (df["p_pick"] >= 0.65).to_numpy()
    report("confident (>=.65)", y[m], p[m])
    report("modest (<.65)", y[~m], p[~m])
    print(f"  {'base rate':<20}{'':>4}  {BASE_RATE_ACC:>6.3f}{'':<12}"
          f"{0.6931:>8.4f}  {0.2500:>7.4f}")
    print("=" * 74)
    n = len(df)
    print(f"{n} bouts. The sd beside each accuracy is the binomial sd: at this n")
    print("a gap smaller than about two sds from the walk-forward number")
    print("(0.6082 +/- 0.0202) is not evidence of anything.")


if __name__ == "__main__":
    main()
