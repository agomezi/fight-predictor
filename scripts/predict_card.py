"""Predict a bout: two names in, a calibrated win probability out.

    python scripts/predict_card.py "Alpha One" "Beta Two"
    python scripts/predict_card.py "Dupe Name" "Heavy Guy" --division Heavyweight
    python scripts/predict_card.py A B --date 2026-09-12 --trees 200

Runs on the STATIC feature set today (reach / height / age / stance), because
src.matchup.build_matchup_row works without a HistoryIndex. Pass --with-history
to add the rolling and Elo differences.

Three things it does that a naive predictor would not:

  * Trains only on fights STRICTLY BEFORE the bout date. For a future bout that
    is everything; for a historical one it is an honest re-enactment rather than
    a model that has already seen the answer.
  * Mirror-averages both corner orderings, so the answer does not depend on
    which fighter was typed first. p = (p(A,B) + 1 - p(B,A)) / 2.
  * Reports SUPPORT. The model will return 0.63 for a debutant as cheerfully as
    for a champion, and the number means far less in the first case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_NAMES, build_feature_table, to_matrix  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.matchup import (  # noqa: E402
    build_training_matrix,
    ROLLING_DIFF_NAMES,
    FighterBios,
    build_matchup_row,
    rows_to_matrix,
)

RANDOM_SEED = 42
THIN_FIGHTS = 3


def prior_fight_counts(features: pd.DataFrame, before) -> dict:
    """How many logged fights each fighter had strictly before `before`."""
    past = features[features["Event_Date"] < pd.Timestamp(before)]
    counts: dict = {}
    for col in ("fighter_A_url", "fighter_B_url"):
        for url, n in past[col].value_counts().items():
            counts[url] = counts.get(url, 0) + int(n)
    return counts


def support_label(n_a: int, n_b: int) -> tuple[str, str]:
    """Coarse support tier plus the reason, from both fighters' history depth."""
    fewest = min(n_a, n_b)
    if fewest == 0:
        return "none", "one fighter has no prior fights in the data"
    if fewest < THIN_FIGHTS:
        return "thin", f"thinnest record is {fewest} prior fight(s)"
    return "ok", f"both fighters have {THIN_FIGHTS}+ prior fights"


def resolve_or_exit(bios: FighterBios, name: str, division, assume_yes=False):
    """Resolve a typed name, offering a confirmable suggestion when it misses.

    Names in the dataset are exact strings ("Ian Machado Garry", not "Ian
    Garry"), and the resolver matches them exactly -- for good reason, since
    guessing silently would predict the wrong fighter. But a flat "no fighter
    named X" makes the tool hostile to use, so a miss now proposes the closest
    matches with each one's record and division, and asks.

    The confirmation is the point: the suggestion is a GUESS, and a prediction
    for the wrong person is worse than an error message. Nothing is auto-picked
    unless --yes says so.
    """
    url, note = bios.resolve_name(name, division)
    if url is not None:
        return url, note

    # Two different misses. An exact name that resolves to several fighters is
    # an AMBIGUITY -- the right options are already known. An unknown name is a
    # TYPO or a partial, so the options have to be guessed.
    exact = bios.candidates(name)
    if exact:
        options = list(exact)
        print(f"'{name}' matches {len(options)} fighters:", file=sys.stderr)
    else:
        options = [u for s in bios.suggest(name) for u in bios.candidates(s)]
        if not options:
            print(f"error: no fighter named {name!r}, and nothing close.",
                  file=sys.stderr)
            print("  Try a surname, or grep the roster:", file=sys.stderr)
            print(f'  grep -i "{name.split()[-1] if name.split() else name}" '
                  "data/ufc_fighters_final.csv", file=sys.stderr)
            sys.exit(2)
        print(f"No fighter named '{name}'. Did you mean:", file=sys.stderr)

    options = options[:6]
    for i, u in enumerate(options, 1):
        print(f"  {i}. {bios.describe(u)}", file=sys.stderr)

    if assume_yes:
        print(f"--yes: using 1. {bios.describe(options[0])}\n", file=sys.stderr)
        return options[0], "auto-accepted suggestion"

    # A piped or redirected stdin cannot answer, and blocking on input() there
    # would hang a script rather than fail it.
    if not sys.stdin.isatty():
        print("(not a terminal -- rerun with the exact name, or pass --yes "
              "to take the first suggestion)", file=sys.stderr)
        sys.exit(2)

    prompt = ("Use 1? [Y/n] " if len(options) == 1
              else f"Which one? [1-{len(options)}, or n to cancel] ")
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.", file=sys.stderr)
        sys.exit(2)

    if answer in ("n", "no", "q"):
        sys.exit(2)
    if answer in ("", "y", "yes"):
        chosen = options[0]
    elif answer.isdigit() and 1 <= int(answer) <= len(options):
        chosen = options[int(answer) - 1]
    else:
        print("not a valid choice; cancelled.", file=sys.stderr)
        sys.exit(2)

    print(f"using {bios.describe(chosen)}\n", file=sys.stderr)
    return chosen, "confirmed suggestion"


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a UFC bout.")
    ap.add_argument("fighter_a")
    ap.add_argument("fighter_b")
    ap.add_argument("--division", default=None,
                    help="weight class, e.g. 'Lightweight'. Inferred from the "
                         "fighters' most recent bouts when omitted.")
    ap.add_argument("--date", default=None,
                    help="bout date (YYYY-MM-DD). Defaults to today.")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--min-train", type=int, default=50,
                    help="refuse to fit on fewer fights than this (default 50). "
                         "Lower it only for smoke-testing against a fixture.")
    ap.add_argument("--with-history", action="store_true",
                    help="add rolling + Elo features")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="accept the top name suggestion without asking")
    args = ap.parse_args()

    when = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    features, _ = build_feature_table(seed=RANDOM_SEED)
    bios = FighterBios()

    # --- division: infer rather than demand, but say what was chosen --------
    division = args.division
    inferred = False
    if division is None:
        a_url, _ = bios.resolve_name(args.fighter_a, None)
        b_url, _ = bios.resolve_name(args.fighter_b, None)
        recent = []
        for url in (a_url, b_url):
            if url is None:
                continue
            hist = features[(features["fighter_A_url"] == url)
                            | (features["fighter_B_url"] == url)]
            if len(hist):
                recent.append(hist.sort_values("Event_Date").iloc[-1]["Weight_Class"])
        division = recent[0] if recent else None
        inferred = division is not None

    url_a, note_a = resolve_or_exit(bios, args.fighter_a, division, args.yes)
    url_b, note_b = resolve_or_exit(bios, args.fighter_b, division, args.yes)
    if url_a == url_b:
        print("error: both names resolved to the same fighter", file=sys.stderr)
        sys.exit(2)

    # Report the CANONICAL names from here on, not what was typed. After a
    # confirmed suggestion the two differ, and every number below describes the
    # fighter that was resolved.
    name_a, name_b = bios.name_of(url_a), bios.name_of(url_b)

    # --- train on the past only --------------------------------------------
    train_df = features[features["Event_Date"] < when]
    if len(train_df) < args.min_train:
        print(f"error: only {len(train_df)} fights before {when.date()}; "
              "not enough to fit on", file=sys.stderr)
        sys.exit(3)

    index = priors = None
    # The model must be TRAINED on the same feature set it is SERVED. With
    # --with-history that means training on the rolling matrix too, not just
    # serving rolling features into a model that never saw them -- otherwise the
    # extra columns are silently ignored and the "static + rolling + Elo" label
    # is a lie. So the column list and the training matrix are chosen together.
    elo_index = None
    if args.with_history:
        from src.history import (
            EloIndex, HistoryIndex, build_event_log, division_priors,
        )
        log, log_info = build_event_log(seed=RANDOM_SEED)
        index, priors = HistoryIndex(log), division_priors(log)
        # Elo is per-fight for training and as-of-date for serving. Both come
        # from one EloIndex, so the two paths cannot disagree. Passing None
        # here is what left elo_diff a constant-zero dead column previously.
        elo_index = EloIndex(log)
        if log_info["stats_missing"]:
            print(f"note: stat columns absent from the raw table: "
                  f"{log_info['stats_missing']}\n")
        X, y, cols = build_training_matrix(train_df, bios, index=index,
                                           priors=priors, elo_index=elo_index)
    else:
        cols = list(FEATURE_NAMES)
        X, y = to_matrix(train_df)

    forest = RandomForest(n_trees=args.trees, max_depth=12, min_samples_split=10,
                          min_samples_leaf=5, feature_subset="sqrt",
                          oob_score=False, random_state=RANDOM_SEED).fit(X, y)

    # --- mirror-average both orderings -------------------------------------
    # Serving ratings: each fighter's rating after every fight strictly before
    # the bout. For a future bout that is simply their current rating.
    serve_elo = elo_index.as_of(when) if elo_index is not None else None
    fwd = build_matchup_row(url_a, url_b, division, when, bios,
                            index=index, priors=priors, elo_ratings=serve_elo)
    rev = build_matchup_row(url_b, url_a, division, when, bios,
                            index=index, priors=priors, elo_ratings=serve_elo)
    p_fwd = float(forest.predict_proba(rows_to_matrix([fwd], cols))[0])
    p_rev = float(forest.predict_proba(rows_to_matrix([rev], cols))[0])
    p = (p_fwd + (1.0 - p_rev)) / 2.0

    counts = prior_fight_counts(features, when)
    n_a, n_b = counts.get(url_a, 0), counts.get(url_b, 0)
    tier, why = support_label(n_a, n_b)

    width = 74
    print("=" * width)
    print(f"{name_a}  vs  {name_b}")
    print("=" * width)
    print(f"division      : {division or 'unknown'}"
          + ("   (inferred from recent bouts -- override with --division)"
             if inferred else ""))
    print(f"bout date     : {when.date()}")
    print(f"trained on    : {len(train_df)} fights before that date")
    # Elo is threaded through the feature set but left at its initial value
    # everywhere (no per-fight elo_ratings), so the elo_diff column is present
    # but inert in both training and serving -- consistent, but not yet a live
    # signal. Say so rather than claim an Elo edge the model has not been given.
    print(f"features      : {'static + rolling form + live Elo' if index else 'static only'}"
          + f"  ({len(cols)} columns)")
    if serve_elo is not None:
        ra, rb = serve_elo.get(url_a), serve_elo.get(url_b)
        show = lambda v: "unrated (debut)" if v is None else f"{v:.0f}"
        print(f"Elo (as of {when.date()}): {name_a} {show(ra)} | "
              f"{name_b} {show(rb)}")
    print()
    print(f"P({name_a} wins) = {p:.3f}")
    print(f"P({name_b} wins) = {1.0 - p:.3f}")
    print(f"  (both orderings: {p_fwd:.3f} forward, {1.0 - p_rev:.3f} reversed; "
          f"spread {abs(p_fwd - (1.0 - p_rev)):.3f})")
    print()
    print(f"support       : {tier.upper()}  -- {why}")
    print(f"                {name_a}: {n_a} prior | "
          f"{name_b}: {n_b} prior")
    if tier != "ok":
        print("                Treat this probability as weakly informed.")
    print()
    print("Calibration caveat: on the static feature set this model scores about")
    print("0.674 log loss against 0.693 for a coin flip -- a real but small edge.")
    print("Probabilities cluster near 0.5 and should. Anything near 0.9 from")
    print("these features would be a bug, not a strong opinion.")
    if not index:
        print("Rolling form and Elo are not in this prediction. Pass "
              "--with-history to include them.")

    # The verdict, in one plain sentence. Everything above is the working; this
    # is the answer, phrased favourite-first so it reads the way a person would
    # say it. Canonical names, not what was typed -- after a confirmed
    # suggestion those differ, and this should name who was actually predicted.
    if p >= 0.5:
        fav, dog, pct = name_a, name_b, p
    else:
        fav, dog, pct = name_b, name_a, 1.0 - p
    print()
    print("-" * width)
    print(f"{fav} has a {pct:.1%} chance of winning against {dog}.")
    if pct < 0.55:
        print("That is close to a coin flip -- treat it as a lean, not a pick.")
    print("-" * width)


if __name__ == "__main__":
    main()
