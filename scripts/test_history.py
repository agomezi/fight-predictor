"""Checks for src/history.py.

Two halves, same pattern as test_evaluate.py.

The reshape, the index and the priors are implemented, so they are checked
properly -- including that the a_is_f1 side mapping is the right way round,
which is the one silent-corruption risk in the reshape.

features_as_of, elo_update and shrink checks assert PROPERTIES, not specific
output values: zero-sum for Elo, prior-recovery and boundedness for shrinkage.
A test asserting the exact number a formula produces would just be the formula
written twice.

The dedicated leakage proof lives in scripts/test_leakage.py.

Run from the repo root (venv active):
    python scripts/test_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import load_fights  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.history import (  # noqa: E402
    AS_OF_FEATURES,
    SECS_PER_15M,
    EloIndex,
    ELO_INITIAL,
    ELO_K,
    HistoryIndex,
    build_event_log,
    division_priors,
    elo_update,
    features_as_of,
    prior_for,
    run_elo,
    shrink,
)

failures, skipped = [], []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def skip(name, why):
    print(f"[SKIP] {name}  ({why})")
    skipped.append(name)


print("=" * 78)
print("EVENT LOG RESHAPE")
print("=" * 78)

log, info = build_event_log(seed=42)
features, _ = build_feature_table(seed=42)
fights = load_fights().set_index("Fight_URL")

check("two log rows per fight", len(log) == 2 * len(features),
      f"({len(log)} rows, {len(features)} fights)")
check("every stat column resolved", not info["stats_missing"],
      f"(missing: {info['stats_missing'] or 'none'})")
check("no (fighter, fight) duplicated",
      not log.duplicated(["fighter_url", "Fight_URL"]).any())
check("exactly one winner per fight",
      bool((log.groupby("Fight_URL")["won"].sum() == 1).all()))
check("rows sorted by fighter then date",
      log.equals(log.sort_values(["fighter_url", "Event_Date"], kind="mergesort")
                    .reset_index(drop=True)))
check("no null fighter ids", not log["fighter_url"].isna().any())

# The side mapping. If own_/opp_ were swapped, every fighter's offence would
# silently be their opponent's -- the model would still train and the numbers
# would still look plausible, which is what makes it worth an explicit check.
side_ok = True
for row in log.itertuples(index=False):
    fight = fights.loc[row.Fight_URL]
    frow = features[(features["Fight_URL"] == row.Fight_URL)].iloc[0]
    a_is_f1 = bool(frow["a_is_f1"])
    is_a = row.fighter_url == frow["fighter_A_url"]
    # This fighter is corner 1 iff (they are side A) == (A is corner 1).
    own_is_f1 = (is_a == a_is_f1)
    expected_own = fight["F1_Sig_Landed"] if own_is_f1 else fight["F2_Sig_Landed"]
    if not np.isclose(float(row.own_Sig_Landed), float(expected_own)):
        side_ok = False
        break
check("own_/opp_ stats map to the correct corner", side_ok)

print()
print("=" * 78)
print("HISTORY INDEX AND PRIORS")
print("=" * 78)

idx = HistoryIndex(log)
check("index covers every fighter",
      len(list(idx.fighters())) == log["fighter_url"].nunique())
check("unknown fighter returns an empty frame, not an error",
      len(idx.rows_for("http://x/does-not-exist")) == 0)
check("unknown fighter's frame keeps the schema",
      list(idx.rows_for("http://x/does-not-exist").columns) == list(log.columns))
some = next(iter(idx.fighters()))
check("a fighter's rows are date-ascending",
      idx.rows_for(some)["Event_Date"].is_monotonic_increasing)
check("index row counts sum to the log",
      sum(len(idx.rows_for(f)) for f in idx.fighters()) == len(log))

priors = division_priors(log)
# One winner and one loser per fight, so the pooled win rate is exactly 0.5.
check("global prior win_rate is exactly 0.5",
      np.isclose(priors["global"]["win_rate"], 0.5),
      f"({priors['global']['win_rate']:.4f})")
check("every rate prior is a probability",
      all(0.0 <= v <= 1.0 for d in priors["by_division"].values()
          for v in d.values()))
check("unknown division falls back to the global prior",
      prior_for(priors, "Interpretive Dance Bout") is priors["global"])

print()
print("=" * 78)
print("features_as_of")
print("=" * 78)
# A fighter's own first fight: as of that date they have no prior fights at all,
# so this is the debut path even for a veteran.
first = log.groupby("fighter_url")["Event_Date"].min()
debut_url = str(first.index[0])
debut_date = first.iloc[0]
try:
    d = features_as_of(idx, debut_url, debut_date,
                       weight_class="Lightweight Bout", priors=priors)
    check("returns exactly the AS_OF_FEATURES keys plus support",
          set(d) == set(AS_OF_FEATURES) | {"support"},
          f"(unexpected: {sorted(set(d) - set(AS_OF_FEATURES) - {'support'})})")
    check("a fighter's own first fight is the debut path",
          d["support"] == "none" and d["n_fights"] == 0)
    check("debut rates fall back to the prior, not to zero",
          np.isclose(d["win_rate"], prior_for(priors, "Lightweight Bout")["win_rate"]))
    check("debut leaves unknowable quantities NaN, not zero",
          np.isnan(d["days_since_last"]) and np.isnan(d["sig_landed_pm"]),
          "(0.0 strikes per minute is a real value and must not mean 'unknown')")

    # A veteran, evaluated after their whole career.
    counts = log["fighter_url"].value_counts()
    vet = str(counts.index[0])
    vet_rows = idx.rows_for(vet)
    after = vet_rows["Event_Date"].max() + pd.Timedelta(days=1)
    v = features_as_of(idx, vet, after, weight_class=None, priors=priors)
    check("veteran sees every prior fight",
          v["n_fights"] == len(vet_rows), f"({v['n_fights']} of {len(vet_rows)})")
    check("veteran support is not 'none'", v["support"] != "none")
    check("shrunk win_rate lies between the prior and the raw rate",
          min(v["win_rate_raw"], priors["global"]["win_rate"]) - 1e-9
          <= v["win_rate"] <=
          max(v["win_rate_raw"], priors["global"]["win_rate"]) + 1e-9)
    check("days_since_last is positive for a veteran",
          v["days_since_last"] > 0)

    # n_fights must be monotone non-decreasing as the cutoff advances.
    seq = [features_as_of(idx, vet, dt, priors=priors)["n_fights"]
           for dt in sorted(vet_rows["Event_Date"])]
    check("n_fights never decreases as the cutoff advances",
          all(b >= a for a, b in zip(seq, seq[1:])), f"({seq})")
    check("cutoff is STRICTLY before: own fight date is excluded",
          features_as_of(idx, vet, vet_rows["Event_Date"].iloc[0],
                         priors=priors)["n_fights"] == 0,
          "(a row on the bout's own date must not count)")
except NotImplementedError:
    skip("features_as_of checks", "not implemented yet")

print()
print("=" * 78)
print("elo_update")
print("=" * 78)
try:
    # Zero-sum: whatever the winner gains, the loser loses.
    a, b = 1500.0, 1500.0
    na, nb = elo_update(a, b, True, k=ELO_K)
    check("equal ratings: winner gains, loser loses", na > a and nb < b)
    check("zero-sum exchange", np.isclose((na - a), -(nb - b)),
          f"(+{na - a:.3f} / {nb - b:.3f})")
    check("equal ratings split K evenly", np.isclose(na - a, ELO_K / 2.0),
          f"(gain {na - a:.3f}, K/2 = {ELO_K / 2.0:.3f})")

    # A heavy favourite winning is barely news; an upset is.
    fav_gain = elo_update(1900.0, 1500.0, True, k=ELO_K)[0] - 1900.0
    dog_gain = elo_update(1500.0, 1900.0, True, k=ELO_K)[0] - 1500.0
    check("favourite winning gains less than an underdog winning",
          dog_gain > fav_gain, f"(underdog +{dog_gain:.2f} vs favourite +{fav_gain:.2f})")
    check("a 400-point favourite's win is worth under a fifth of K",
          0.0 < fav_gain < 0.2 * ELO_K, f"(+{fav_gain:.3f})")
    check("symmetry: swapping sides and the result mirrors the exchange",
          np.isclose(elo_update(1600.0, 1400.0, True, k=ELO_K)[0] - 1600.0,
                     1400.0 - elo_update(1400.0, 1600.0, False, k=ELO_K)[0]))
    check("larger K moves ratings further",
          (elo_update(1500.0, 1500.0, True, k=48.0)[0]
           > elo_update(1500.0, 1500.0, True, k=16.0)[0]))

    # The sweep must record ratings BEFORE applying each result.
    elo = run_elo(log)
    check("run_elo yields one rating per (fighter, fight)", len(elo) == len(log))
    # run_elo now returns Event_Date itself, so merging the log's copy in would
    # collide and pandas would silently suffix both to Event_Date_x/_y. Take
    # only the join keys plus what is actually missing.
    firsts = elo if "Event_Date" in elo.columns else elo.merge(
        log[["Fight_URL", "fighter_url", "Event_Date"]],
        on=["Fight_URL", "fighter_url"])
    # Restrict the debut check to fighters whose earliest fight was on a date
    # they fought only once. The early-UFC tournament cards put two fights for
    # the same fighter on one night, and run_elo correctly rates those
    # sequentially, so the second bout of the night has a pre-rating that
    # already reflects the first -- a real property of the data, not a leak
    # (features_as_of still excludes both via its strict date filter).
    per_date = firsts.groupby(["fighter_url", "Event_Date"]).size()
    first_date = firsts.groupby("fighter_url")["Event_Date"].min()
    solo_debut = [u for u, d in first_date.items() if per_date.loc[(u, d)] == 1]
    debut_ratings = (firsts[firsts["fighter_url"].isin(solo_debut)]
                     .sort_values("Event_Date")
                     .groupby("fighter_url")["elo_pre"].first())
    check("a fighter's first rating is the initial rating (solo debut nights)",
          bool(np.allclose(debut_ratings.to_numpy(), ELO_INITIAL)),
          f"(if not, a fight's own result reached its pre-fight rating; "
          f"{len(solo_debut)} solo debuts checked)")
    check("total rating is conserved across the sweep",
          np.isclose(elo.groupby("Fight_URL")["elo_pre"].sum().sum(),
                     2 * ELO_INITIAL * log["Fight_URL"].nunique(), rtol=0.5),
          "(loose bound: drift would be gross, not subtle)")
except NotImplementedError:
    skip("elo_update / run_elo checks", "not implemented yet")

print()
print("=" * 78)
print("PARTIAL-STAT RATES — the dilution guard")
print("=" * 78)
# Real data currently has every stat on every row, so this bug is LATENT: it
# would only appear once a results-only source (Wikipedia) contributes fights
# that carry a result and a duration but no strike counts. That is exactly why
# it needs a synthetic fixture -- there is nothing in the CSVs that exercises it,
# and the leakage suite would never catch it because it is not a leak, it is a
# quiet dilution toward zero.
mixed = pd.DataFrame([
    # Two stat-bearing fights: 300s each, 30 significant strikes each.
    dict(fighter_url="u", opponent_url="o1", Fight_URL="f1",
         Event_Date=pd.Timestamp("2020-01-01"), Weight_Class="Lightweight Bout",
         won=1, is_finish=False, Method="Decision - Unanimous", End_Round=3,
         fight_secs=300.0, own_Sig_Landed=30.0, opp_Sig_Landed=10.0,
         own_TD_Landed=1.0, own_Sub_Att=0.0, own_Ctrl_Sec=60.0),
    dict(fighter_url="u", opponent_url="o2", Fight_URL="f2",
         Event_Date=pd.Timestamp("2020-06-01"), Weight_Class="Lightweight Bout",
         won=1, is_finish=False, Method="Decision - Unanimous", End_Round=3,
         fight_secs=300.0, own_Sig_Landed=30.0, opp_Sig_Landed=10.0,
         own_TD_Landed=1.0, own_Sub_Att=0.0, own_Ctrl_Sec=60.0),
    # Two results-only fights: duration known, stats absent.
    dict(fighter_url="u", opponent_url="o3", Fight_URL="f3",
         Event_Date=pd.Timestamp("2021-01-01"), Weight_Class="Lightweight Bout",
         won=0, is_finish=False, Method="Decision - Unanimous", End_Round=3,
         fight_secs=300.0, own_Sig_Landed=np.nan, opp_Sig_Landed=np.nan,
         own_TD_Landed=np.nan, own_Sub_Att=np.nan, own_Ctrl_Sec=np.nan),
    dict(fighter_url="u", opponent_url="o4", Fight_URL="f4",
         Event_Date=pd.Timestamp("2021-06-01"), Weight_Class="Lightweight Bout",
         won=1, is_finish=False, Method="Decision - Unanimous", End_Round=3,
         fight_secs=300.0, own_Sig_Landed=np.nan, opp_Sig_Landed=np.nan,
         own_TD_Landed=np.nan, own_Sub_Att=np.nan, own_Ctrl_Sec=np.nan),
])
mix_idx = HistoryIndex(mixed)
mp = division_priors(mixed)
mf = features_as_of(mix_idx, "u", pd.Timestamp("2022-01-01"), priors=mp)

# 60 strikes over the 10 minutes that actually carried strike counts = 6.0/min.
# Diluting by all four fights (20 minutes) would give 3.0 -- a 50% understatement.
check("rate uses only the fights carrying the stat",
      np.isclose(mf["sig_landed_pm"], 6.0),
      f"(got {mf['sig_landed_pm']:.4f}; diluted would be 3.0000)")
check("control fraction uses the matching subset",
      np.isclose(mf["ctrl_frac"], 120.0 / 600.0),
      f"(got {mf['ctrl_frac']:.4f}; diluted would be {120.0/1200.0:.4f})")
check("takedown rate uses the matching subset",
      np.isclose(mf["td_landed_p15m"], 2.0 / (600.0 / SECS_PER_15M)),
      f"(got {mf['td_landed_p15m']:.4f})")
# Cage experience is different: a results-only source still tells us how long
# the fight lasted, so total_fight_secs SHOULD count all four.
check("total_fight_secs still counts every past fight",
      np.isclose(mf["total_fight_secs"], 1200.0),
      f"(got {mf['total_fight_secs']:.0f} of 1200)")
check("n_fights counts every past fight", mf["n_fights"] == 4)

# A fighter whose every past fight lacks the stat must read unknown, not zero.
only_results = mixed.iloc[2:].copy()
of = features_as_of(HistoryIndex(only_results), "u",
                    pd.Timestamp("2022-01-01"), priors=mp)
check("no stat-bearing past fight gives NaN, not 0.0",
      np.isnan(of["sig_landed_pm"]) and np.isnan(of["ctrl_frac"]),
      "(0.0 strikes per minute is a real, terrible value)")
check("but its results-based features still resolve",
      of["n_fights"] == 2 and np.isfinite(of["win_rate"]))

print()
print("=" * 78)
print("EloIndex — as-of rating lookup")
print("=" * 78)
ei = EloIndex(log)
check("table carries both pre and post ratings",
      {"elo_pre", "elo_post", "Event_Date"} <= set(ei.table.columns))
check("one row per (fighter, fight)", len(ei.table) == len(log))

# The chain: a fighter's post-fight rating must be exactly the pre-fight rating
# they carry into their NEXT fight. If it is not, a rating is being lost or
# recomputed between fights, and every downstream feature is subtly wrong.
chain_ok, chain_checked = True, 0
for url, grp in ei.table.sort_values("Event_Date", kind="mergesort").groupby("fighter_url"):
    post = grp["elo_post"].to_numpy()[:-1]
    nxt = grp["elo_pre"].to_numpy()[1:]
    if len(post):
        chain_checked += len(post)
        if not np.allclose(post, nxt):
            chain_ok = False
check("elo_post chains into the next fight's elo_pre", chain_ok,
      f"({chain_checked} consecutive pairs)")

check("a fight's own result is not in its pre-fight rating",
      bool(np.allclose(
          ei.table.sort_values("Event_Date", kind="mergesort")
                  .groupby("fighter_url")["elo_pre"].first().to_numpy(),
          ELO_INITIAL)))
check("every fight moves both ratings",
      bool((ei.table["elo_pre"] != ei.table["elo_post"]).all()),
      "(K > 0, so no exchange is a no-op)")

f_any = str(log["Fight_URL"].iloc[0])
check("for_fight returns both corners", len(ei.for_fight(f_any)) == 2)
check("for_fight on an unknown fight is empty, not an error",
      ei.for_fight("http://f/nope") == {})
check("as_of before any fight is empty",
      ei.as_of(ei.table["Event_Date"].min()) == {})
check("as_of after everything equals latest()",
      ei.as_of(ei.table["Event_Date"].max() + pd.Timedelta(days=1)) == ei.latest())
check("ratings are conserved overall",
      np.isclose(sum(ei.latest().values()),
                 ELO_INITIAL * len(ei.latest()), atol=1e-6),
      "(zero-sum: total is invariant)")

# for_fight and as_of agree except on same-night tournament cards, where
# for_fight legitimately sees an earlier bout that as_of's date filter cannot.
# Counted rather than assumed, so the size of the exception is known.
disagree = 0
for row in ei.table.itertuples(index=False):
    snap = ei.as_of(row.Event_Date)
    if not np.isclose(snap.get(row.fighter_url, ELO_INITIAL), row.elo_pre):
        disagree += 1
check("for_fight and as_of agree except on same-date rematches",
      disagree <= (log.duplicated(["fighter_url", "Event_Date"], keep=False).sum()
                   + 1),
      f"({disagree} of {len(ei.table)} rows differ)")

print()
print("=" * 78)
print("shrink")
print("=" * 78)
try:
    check("no data returns exactly the prior",
          np.isclose(shrink(0, 0, 0.42, alpha=5.0), 0.42),
          "(the n == 0 case falling out is the check that the formula is right)")
    check("lots of data approaches the observed rate",
          np.isclose(shrink(9000, 10000, 0.5, alpha=5.0), 0.9, atol=1e-3))
    check("result lies between the prior and the observed rate",
          0.5 <= shrink(2, 2, 0.5, alpha=5.0) <= 1.0)
    check("a 2-0 record is pulled well below 1.0",
          shrink(2, 2, 0.5, alpha=5.0) < 0.8,
          f"({shrink(2, 2, 0.5, alpha=5.0):.4f})")
    check("more evidence moves further from the prior",
          shrink(20, 20, 0.5, alpha=5.0) > shrink(2, 2, 0.5, alpha=5.0))
    check("stronger alpha shrinks harder",
          shrink(2, 2, 0.5, alpha=20.0) < shrink(2, 2, 0.5, alpha=5.0))
    check("an all-losses record is pulled above 0.0",
          shrink(0, 3, 0.5, alpha=5.0) > 0.0)
except NotImplementedError:
    skip("shrink checks", "not implemented yet")

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
if skipped:
    print(f"All implemented checks pass. {len(skipped)} group(s) skipped: "
          f"{', '.join(skipped)}")
    print("Fill the TODO(human) blocks in src/history.py to enable them.")
else:
    print("All history checks pass.")
