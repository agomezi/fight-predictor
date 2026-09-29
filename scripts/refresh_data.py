"""Append newly-completed UFC events to the fights CSV, from ESPN.

The model's data ends at the last ingest while cards keep happening, so every
prediction made after that is served with Elo and rolling form frozen. That is a
plumbing problem, not a modelling one, and it is the cheapest accuracy
available.

WHAT THIS REFRESHES. All 14 rolling features. That is the change: this script
used to scrape Wikipedia, which carries results but no per-fight stats, so
sig_landed_pm, sig_absorbed_pm, td_landed_p15m, sub_att_p15m and ctrl_frac
stayed frozen for any fighter who had fought since the last ingest. ESPN
publishes a full box score per bout, so those five now refresh too. See
src/espn.py for the endpoint and for the three measured gotchas behind it.

WHY NOT ufcstats. It publishes the same five but now serves a JavaScript
proof-of-work interstitial; getting past that programmatically would be
circumventing bot detection. WHY NOT ufc.com/athlete: it publishes four of the
five as CAREER AVERAGES AS OF TODAY, and joining those onto a historical row
backfills a fighter's future into their past -- exactly the leakage the
chronological split exists to prevent.

CONTROL TIME BEFORE 2015 IS WRITTEN EMPTY, NOT ZERO. ESPN reports untracked
control time as 0:00, which is indistinguishable from a real zero. The
missing-flag machinery in features_as_of handles absence correctly; a zero would
read as a measurement of nothing. See CTRL_TRACKED_FROM in src/espn.py.

Any stat column ESPN omits is likewise written EMPTY, never zero, and
src/history._rate_basis computes each rate over only the fights that carry it,
so partial data dilutes nothing.

SAFETY. Dry-run is the default: this prints what it would do and writes nothing
unless --write is passed, and even then it writes a NEW file and leaves the
original untouched. Validation runs before any write and aborts on the first
violation.

    python scripts/refresh_data.py                    # dry run, all new events
    python scripts/refresh_data.py --limit 2          # dry run, first 2 only
    python scripts/refresh_data.py --write            # actually write

After writing, run the gate: test_leakage.py, test_history.py,
test_matchup.py, then evaluate_models.py. Walk-forward accuracy must stay inside
a sane band of 0.6082 +/- 0.0202. A jump to 0.70 is a bug report, not a win.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_loading import FIGHTERS_CSV, FIGHTS_CSV, load_fights  # noqa: E402
from src.espn import (  # noqa: E402
    CTRL_TRACKED_FROM,
    POLITE_DELAY_S,
    extract_corner_stats,
    fetch_card,
    fetch_round_formats,
    is_ufc_event,
    iter_competitions,
    list_events,
    parse_clock_seconds,
    weight_class_string,
)

OUT_DEFAULT = REPO / "data" / "ufc_gold_dataset_refreshed.csv"

# The fights-table schema, needed by --since mode where no CSV is present to
# read a header from. Kept in the same order as the source file.
SCHEMA_COLUMNS = (
    "Fight_URL", "Fighter_1", "Fighter_2", "Winner", "Weight_Class", "Method",
    "End_Round", "End_Time", "Total_Fight_Time_Sec", "Time_Format",
    "F1_KD", "F2_KD", "F1_Sig_Landed", "F1_Sig_Att", "F2_Sig_Landed",
    "F2_Sig_Att", "F1_TD_Landed", "F2_TD_Landed", "F1_TD_Att", "F2_TD_Att",
    "F1_Sub_Att", "F2_Sub_Att", "F1_Ctrl_Sec", "F2_Ctrl_Sec",
    "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg",
    "F1_Distance", "F2_Distance", "F1_Clinch", "F2_Clinch", "F1_Ground",
    "F2_Ground", "Event_Date",
)

# Letters that NFKD does not decompose, because the diacritic is part of the
# glyph rather than a combining mark. Without these, the stripping step below
# DELETES them instead of folding them: "Syguła" -> "sygua", which then fails to
# match the bios' "Sygula" and silently creates a second fighter with no
# history. Found in the first ESPN ingest, where it split a fighter with three
# prior UFC bouts.
UNDECOMPOSED = {
    "ł": "l", "Ł": "L",   # Polish l with stroke
    "ø": "o", "Ø": "O",   # Scandinavian o with stroke
    "đ": "d", "Đ": "D",   # d with stroke (Croatian, Vietnamese)
    "ħ": "h", "Ħ": "H",   # h with stroke (Maltese)
    "ß": "ss",                  # German sharp s
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "þ": "th", "Þ": "Th",
}

# Suffixes that distinguish a fighter from a RELATIVE, not from themselves.
# Deliberately NOT stripped. "Jessie Rosas" and "Jesse Rosas Jr." are brothers,
# and folding the suffix away would merge two people into one record --
# strictly worse than the split it would fix.
NAME_SUFFIX_NOTE = "Jr./Sr./III are load-bearing; see canonicalise()"


def fold(name: str) -> str:
    """Accent- and case-insensitive key, for matching names across sources.

    Maps undecomposable stroked letters explicitly BEFORE stripping, so they
    fold rather than vanish. Suffixes are left alone on purpose -- see
    NAME_SUFFIX_NOTE.
    """
    text = name or ""
    for glyph, plain in UNDECOMPOSED.items():
        text = text.replace(glyph, plain)
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", ascii_only.lower()).strip()


def synth_fight_url(date_iso: str, a: str, b: str) -> str:
    """A stable synthetic id. Must be identical across runs, or validation
    would see the same bout as new every week and duplicate it."""
    slug = "-".join(sorted([fold(a).replace(" ", "_"), fold(b).replace(" ", "_")]))
    return f"espn:{date_iso}:{slug}"


def discover_new_events(after: pd.Timestamp, limit=None):
    """(date, event_id, event_name) for completed events dated after `after`.

    ESPN's event index is per calendar year, so this walks the years the window
    touches rather than pulling the whole history. Scheduled-but-not-yet-fought
    cards are filtered later, in build_rows, where the bout status is visible.
    """
    today = pd.Timestamp.today().normalize()
    found = []
    for year in range(after.year, today.year + 1):
        for event_id, date_iso, name in list_events(year):
            when = pd.to_datetime(date_iso, errors="coerce")
            if pd.isna(when) or when <= after or when > today or not name:
                continue
            # ESPN files Contender Series and Road to UFC under the UFC league.
            # Those are not UFC bouts and the existing dataset contains none of
            # them -- verified: Contender Series dates hold 0 fights while UFC
            # cards on neighbouring dates hold full slates.
            if not is_ufc_event(name):
                continue
            found.append((when.normalize(), event_id, name))
    found = sorted(set(found))
    return found[:limit] if limit else found


def build_rows(events, columns, canon=None, verbose=True):
    """Turn ESPN cards into CSV rows matching the existing schema.

    Returns (rows, failed) where `failed` names every event that produced no
    bouts. Reported loudly at the end rather than as a line that scrolls past:
    a partial ingest that looks successful is how the four biggest cards of a
    five-month catch-up went missing once already.
    """
    canon = canon or {}
    rows, failed = [], []
    for when, event_id, name in events:
        date_iso = when.date().isoformat()
        # Control time is not tracked in the old era and ESPN reports the gap as
        # 0:00. Decided per event, since only the event knows its own date.
        ctrl_tracked = date_iso >= CTRL_TRACKED_FROM
        try:
            card = fetch_card(event_id)
            round_formats = fetch_round_formats(event_id)
        except RuntimeError as exc:
            print(f"  !! {name}: fetch failed ({exc}) -- skipped")
            failed.append(name)
            continue

        bouts = 0
        for segment, competition in iter_competitions(card):
            status = competition.get("status") or {}
            if not ((status.get("type") or {}).get("completed")):
                continue                       # scheduled, or still in progress
            result = status.get("result") or {}

            corners = competition.get("competitors") or []
            if len(corners) != 2:
                continue
            # order 1 is ESPN's first-listed corner. Sorted explicitly rather
            # than trusted, so F1/F2 is stable across runs -- synth_fight_url
            # sorts its own inputs, but the stat columns do not.
            corners = sorted(corners, key=lambda c: c.get("order", 0))

            names = []
            for corner in corners:
                athlete = corner.get("athlete") or {}
                names.append(canonicalise(athlete.get("displayName", ""), canon))
            if not all(names):
                continue

            winners = [n for n, c in zip(names, corners) if c.get("winner")]
            # A draw or no-contest has no single winner. Written blank and
            # reported rather than guessed; validate() rejects the row, which is
            # the honest outcome for a bout with no winner to predict.
            winner = winners[0] if len(winners) == 1 else ""

            end_round = status.get("period") or ""
            end_time = status.get("displayClock") or ""
            elapsed = parse_clock_seconds(end_time)
            secs = ("" if elapsed is None or not end_round
                    else (int(end_round) - 1) * 300 + elapsed)

            row = {c: "" for c in columns}
            row.update({
                "Fight_URL": synth_fight_url(date_iso, names[0], names[1]),
                "Fighter_1": names[0],
                "Fighter_2": names[1],
                "Winner": winner,
                "Weight_Class": weight_class_string(competition),
                "Method": result.get("displayName", ""),
                "End_Round": end_round,
                "End_Time": end_time,
                "Total_Fight_Time_Sec": secs,
                # Stated outright by ESPN, never inferred from card position.
                # competitions[] is not ordered by card position, so the old
                # "first bout is the main event, therefore five rounds" rule
                # would stamp 5 Rnd onto an early prelim.
                "Time_Format": round_formats.get(competition.get("id"), ""),
                "Event_Date": date_iso,
            })

            for prefix, corner in (("F1", corners[0]), ("F2", corners[1])):
                stats = extract_corner_stats(corner)
                for column, value in stats.items():
                    if column == "ctrl_seconds":
                        continue
                    row[f"{prefix}_{column}"] = value
                seconds = stats.get("ctrl_seconds")
                row[f"{prefix}_Ctrl_Sec"] = (
                    "" if not ctrl_tracked or seconds is None else seconds)

            rows.append(row)
            bouts += 1

        if not bouts:
            print(f"  !! {name}: no completed bouts found -- skipped "
                  "(event may be scheduled, not completed)")
            failed.append(name)
        elif verbose:
            print(f"  {date_iso}  {bouts:>2} bouts  {name}")
        time.sleep(POLITE_DELAY_S)
    return rows, failed


# Hand-verified aliases: a fighter the source spells differently from the bios
# in a way no general rule can safely fix. Each entry was checked against the
# bios record (DOB, height, division, prior bouts) before being added.
#
# WHY NOT A RULE. The obvious rule -- strip Jr./Sr./III -- is unsafe: 13 base
# names in the bios are shared by two rows, and some of those are RELATIVES, not
# spellings. "Jessie Rosas" and "Jesse Rosas Jr." are brothers who have both
# fought in the UFC, so a suffix-stripping rule would merge two people into one
# record. That is a worse error than the split it fixes, and it would be
# invisible afterwards. So this stays a short, explicit, auditable list.
NAME_ALIASES = {
    # ESPN spelling            -> bios spelling
    "Michael Aswell": "Michael Aswell Jr.",
}


def canonicalise(name: str, canon: dict) -> str:
    """Map a scraped name onto the bios CSV's exact spelling when possible.

    resolve_join in src/features.py matches names EXACTLY, so "Borislav Nikolic"
    in the bios and "Borislav Nikolic" with a diacritic from ESPN are two
    different fighters as far as the pipeline is concerned, and the bout gets
    dropped. Folding accents and case to find the existing spelling recovers
    those without touching the core matcher, which the existing 8,400 fights
    depend on.
    """
    name = NAME_ALIASES.get(name, name)
    return canon.get(fold(name), name)


def validate(existing: pd.DataFrame, new_rows, known_names) -> list:
    """Every check that must hold before anything is written."""
    problems = []
    if not new_rows:
        problems.append("no new rows parsed")
        return problems

    prev_max = existing["Event_Date"].max()
    seen = set(existing["Fight_URL"].astype(str))
    ids = [r["Fight_URL"] for r in new_rows]

    if len(ids) != len(set(ids)):
        problems.append("duplicate Fight_URL within the new rows")
    overlap = seen & set(ids)
    if overlap:
        problems.append(f"{len(overlap)} Fight_URL already present in the CSV")
    for r in new_rows:
        when = pd.to_datetime(r["Event_Date"])
        if when <= prev_max:
            problems.append(f"row dated {r['Event_Date']} is not after "
                            f"{prev_max.date()}")
            break
    for r in new_rows:
        if r["Winner"] not in (r["Fighter_1"], r["Fighter_2"]):
            problems.append(f"Winner not one of the two fighters: {r['Winner']}")
            break
    for r in new_rows:
        if not r["Weight_Class"] or not r["Method"]:
            problems.append(f"empty Weight_Class or Method for {r['Fight_URL']}")
            break

    # Name reconciliation is reported, never silently dropped.
    fresh = {n for r in new_rows for n in (r["Fighter_1"], r["Fighter_2"])}
    unknown = sorted(n for n in fresh if fold(n) not in known_names)
    if unknown:
        print(f"\n  {len(unknown)} fighter name(s) with no bio row:")
        for n in unknown[:25]:
            print(f"    - {n}")
        if len(unknown) > 25:
            print(f"    ... and {len(unknown) - 25} more")
        print("  These resolve to no Fighter_URL, so build_feature_table will")
        print("  DROP their bouts (n_dropped_unresolved). Add bio stubs with NaN")
        print("  biometrics to keep them -- the missing-flag path handles it.")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="actually write the output file (default: dry run)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N new events")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--since", default=None,
                    help="discover events after this date (YYYY-MM-DD) instead "
                         "of reading the local CSV's max. Lets CI run discovery "
                         "with no dataset present -- see --pending-only.")
    ap.add_argument("--pending-only", default=None,
                    help="write ONLY the newly-parsed rows to this path and "
                         "skip the merge. The output is small enough to review "
                         "in a pull request diff, which is the point: a "
                         "scheduled job should show a human what it found "
                         "rather than rewrite the training data unattended.")
    ap.add_argument("--add-bio-stubs", action="store_true",
                    help="also write a fighters CSV with NaN-biometric stubs "
                         "for genuinely new fighters, so their bouts are not "
                         "dropped by the name join")
    args = ap.parse_args()

    # --since lets discovery run without the dataset, which is what makes a
    # CI job possible at all: data/ is gitignored, so a runner has no CSVs.
    data_free = args.since is not None
    if data_free:
        prev_max = pd.Timestamp(args.since)
        columns = list(SCHEMA_COLUMNS)
        existing = pd.DataFrame({"Event_Date": [prev_max],
                                 "Fight_URL": ["(none)"]})
        bios_all = pd.DataFrame(columns=["Fighter_Name", "Fighter_URL"])
    else:
        existing = load_fights()
        columns = list(pd.read_csv(FIGHTS_CSV, nrows=0).columns)
        prev_max = existing["Event_Date"].max()
        bios_all = pd.read_csv(FIGHTERS_CSV, dtype=str)
    known = {fold(n) for n in bios_all["Fighter_Name"].astype(str)}
    # Folded key -> the exact spelling already in the bios, so scraped names can
    # be snapped onto it instead of being treated as new fighters.
    canon = {}
    for n in bios_all["Fighter_Name"].astype(str):
        canon.setdefault(fold(n), n)

    if data_free:
        print(f"discovery mode: looking for events after {prev_max.date()} "
              "(no local dataset required)")
    else:
        print(f"local data ends {prev_max.date()}  ({len(existing)} fights)")
    events = discover_new_events(prev_max, args.limit)
    if not events:
        print("nothing new. Up to date.")
        return
    print(f"{len(events)} new event(s) to ingest"
          + (f" (limited to {args.limit})" if args.limit else "") + ":\n")

    new_rows, failed = build_rows(events, columns, canon=canon)
    ingested = len(events) - len(failed)
    print(f"\nparsed {len(new_rows)} bouts from {ingested} of {len(events)} event(s)")
    if failed:
        print(f"\n  {len(failed)} EVENT(S) PRODUCED NOTHING:")
        for name in failed:
            print(f"    - {name}")
        print("  Check each by hand before trusting this run. An event with no")
        print("  results table is usually scheduled-not-yet-fought, which is")
        print("  fine; anything else is a parser problem.")

    problems = validate(existing, new_rows, known) if not data_free else (
        [] if new_rows else ["no new rows parsed"])
    if problems:
        print("\nVALIDATION FAILED -- nothing written:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nvalidation passed: ids unique and unseen, dates after the previous "
          "max, winner is a participant, required fields present")

    stat_cols = [c for c in columns if c.startswith(("F1_", "F2_"))]
    filled = sum(1 for c in stat_cols
                 if any(r.get(c) != "" for r in new_rows))
    print(f"{filled} of {len(stat_cols)} per-fight stat columns carry values "
          "-- all 14 rolling features refresh, including the five that stayed "
          "frozen under the Wikipedia source")
    blank_ctrl = sum(1 for r in new_rows if r.get("F1_Ctrl_Sec") == "")
    if blank_ctrl:
        print(f"{blank_ctrl} row(s) have EMPTY control time (not zero): "
              f"ESPN does not track it before {CTRL_TRACKED_FROM}")

    if args.pending_only:
        out = Path(args.pending_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            w.writerows(new_rows)
        print(f"\nwrote {len(new_rows)} pending rows -> {out}")
        print("This file is the REVIEWABLE unit: small enough to read in a pull")
        print("request diff. Merging it does not change the training data --")
        print("run the full refresh locally, where the CSVs live, to do that.")
        return

    if not args.write:
        print(f"\nDRY RUN. Would append {len(new_rows)} rows -> "
              f"{Path(args.out).name}  ({len(existing) + len(new_rows)} total)")
        print("Re-run with --write to produce the file. The original CSV is")
        print("never modified either way.")
        return

    if args.add_bio_stubs:
        fresh = {n for r in new_rows for n in (r["Fighter_1"], r["Fighter_2"])}
        stubs = sorted(n for n in fresh if fold(n) not in known)
        if stubs:
            bio_out = Path(args.out).with_name("ufc_fighters_refreshed.csv")
            stub_rows = []
            for n in stubs:
                # Biometrics blank, NOT zero. The 0.0/0% pattern in the source
                # data is exactly the trap PROVENANCE.md warns about: a blank
                # reads as unknown and gets a missing flag; a zero reads as a
                # real measurement of nothing.
                row = {c: "" for c in bios_all.columns}
                row["Fighter_Name"] = n
                row["Fighter_URL"] = f"espn:fighter:{fold(n).replace(' ', '_')}"
                stub_rows.append(row)
            combined = pd.concat(
                [bios_all, pd.DataFrame(stub_rows, columns=bios_all.columns)],
                ignore_index=True,
            )
            combined.to_csv(bio_out, index=False)
            print(f"wrote {bio_out.name}  ({len(bios_all)} + {len(stub_rows)} "
                  "stubs). Biometrics blank, not zero.")

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for _, r in pd.read_csv(FIGHTS_CSV, dtype=str).iterrows():
            writer.writerow({c: r.get(c, "") for c in columns})
        writer.writerows(new_rows)
    print(f"\nwrote {out}  ({len(existing) + len(new_rows)} rows)")
    print("The original ufc_gold_dataset_final.csv is untouched. To adopt:")
    print("  1. run the gate (test_leakage, test_history, test_matchup)")
    print("  2. evaluate_models.py -- accuracy must stay near 0.6082 +/- 0.0202")
    print("  3. only then swap the filename in src/data_loading.py")


if __name__ == "__main__":
    main()
