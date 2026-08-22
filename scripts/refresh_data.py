"""Append newly-completed UFC events to the fights CSV, from Wikipedia.

The model's data ends 2026-03-07 while today is later, so every prediction is
made with Elo and rolling form frozen at March. That is a plumbing problem, not
a modelling one, and it is the cheapest accuracy available.

WHAT THIS CAN AND CANNOT REFRESH. Wikipedia event pages carry
    Weight class | Winner | def. | Loser | Method | Round | Time | Notes
and nothing else. So of the 14 rolling features, 9 refresh -- n_fights,
win_rate, win_rate_raw, last3_win_rate, finish_rate, finished_rate,
days_since_last, total_fight_secs, elo -- and 5 do not: sig_landed_pm,
sig_absorbed_pm, td_landed_p15m, sub_att_p15m, ctrl_frac. Those need per-fight
stats, which only ufcstats publishes, and ufcstats now serves a JavaScript
proof-of-work interstitial. The five stay frozen at their March values for
fighters who have fought since. That is a real limitation and belongs in the
README, not buried here.

The stat columns are written EMPTY, never zero. Zero significant strikes is a
real and terrible value; the missing-flag machinery in features_as_of handles
absence correctly, and src/history._rate_basis now computes each rate over only
the fights that carry it, so partial data dilutes nothing.

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
import html
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.scrape_event_context import fetch, strip_tags  # noqa: E402

from src.data_loading import FIGHTERS_CSV, FIGHTS_CSV, load_fights  # noqa: E402

WIKI = "https://en.wikipedia.org/wiki/"
LIST_PAGE = WIKI + "List_of_UFC_events"
OUT_DEFAULT = REPO / "data" / "ufc_gold_dataset_refreshed.csv"
POLITE_DELAY_S = 2.0

# Wikipedia's method strings -> the CSV's vocabulary. Order matters: the
# doctor's-stoppage check must precede the generic TKO one.
METHOD_MAP = (
    (r"doctor", "TKO - Doctor's Stoppage"),
    (r"decision.*unanimous", "Decision - Unanimous"),
    (r"decision.*split", "Decision - Split"),
    (r"decision.*majority", "Decision - Majority"),
    (r"submission", "Submission"),
    (r"\btko\b", "KO/TKO"),
    (r"\bko\b", "KO/TKO"),
    (r"disqualification|\bdq\b", "DQ"),
    (r"no contest|\bnc\b", "NC"),
    (r"could not continue", "Could Not Continue"),
    (r"overturned", "Overturned"),
)


def normalise_method(raw: str) -> str:
    low = (raw or "").lower()
    for pattern, out in METHOD_MAP:
        if re.search(pattern, low):
            return out
    return raw.strip()


def clean_fighter(name: str) -> str:
    """Strip the champion marker and footnote residue from a name cell."""
    name = re.sub(r"\(c\)|\(ic\)", "", name or "")
    name = re.sub(r"\[[a-z0-9]+\]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def fold(name: str) -> str:
    """Accent- and case-insensitive key, for matching names across sources."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", ascii_only.lower()).strip()


def weight_class_string(raw: str, is_title: bool) -> str:
    """Match the CSV's conventions exactly.

    Non-title: "Lightweight Bout". Title: "UFC Lightweight Title Bout".
    Women's divisions keep their prefix, which is what the is_womens_bout flag
    in src/features.py reads.
    """
    div = re.sub(r"\s+", " ", (raw or "")).strip()
    if not div:
        return ""
    if is_title:
        return f"UFC {div} Title Bout"
    return f"{div} Bout"


def total_seconds(end_round: int, end_time: str) -> int:
    """Elapsed fight time. Rounds are five minutes throughout the modern era."""
    try:
        mm, ss = end_time.split(":")
        return (int(end_round) - 1) * 300 + int(mm) * 60 + int(ss)
    except (ValueError, AttributeError):
        return 0


def synth_fight_url(date_iso: str, a: str, b: str) -> str:
    """A stable synthetic id. Must be identical across runs, or validation
    would see the same bout as new every week and duplicate it."""
    slug = "-".join(sorted([fold(a).replace(" ", "_"), fold(b).replace(" ", "_")]))
    return f"wiki:{date_iso}:{slug}"


def discover_new_events(after: pd.Timestamp, limit=None):
    """(date, event_name) for events on the list page dated after `after`."""
    page = fetch(LIST_PAGE)
    found = []
    for table in re.findall(
        r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>.*?</table>', page, re.S
    ):
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [strip_tags(c)
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
            if len(cells) < 3:
                continue
            name, date_raw = cells[1], cells[2]
            when = pd.to_datetime(date_raw, errors="coerce")
            if pd.isna(when) or when <= after or not name:
                continue
            found.append((when.normalize(), name))
    found = sorted(set(found))
    return found[:limit] if limit else found


def parse_event(event_name: str):
    """Fetch one event page and return its bout rows, main event first."""
    slug = event_name.replace(" ", "_")
    page = fetch(WIKI + slug)
    tables = re.findall(
        r'<table[^>]*class="[^"]*(?:toccolours|wikitable)[^"]*"[^>]*>.*?</table>',
        page, re.S,
    )
    bouts = []
    for table in tables:
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [strip_tags(c)
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
            rows.append(cells)
        if not any(len(r) >= 7 and r[2].lower().startswith("def") for r in rows):
            continue
        for cells in rows:
            if len(cells) < 7 or not cells[2].lower().startswith("def"):
                continue
            division, winner_raw, _, loser_raw = cells[0], cells[1], cells[2], cells[3]
            method_raw, rnd, tm = cells[4], cells[5], cells[6]
            title = "(c)" in winner_raw or "(c)" in loser_raw
            bouts.append({
                "division": division,
                "winner": clean_fighter(winner_raw),
                "loser": clean_fighter(loser_raw),
                "method": normalise_method(method_raw),
                "round": rnd.strip(),
                "time": tm.strip(),
                "is_title": title,
            })
        if bouts:
            break
    return bouts


def canonicalise(name: str, canon: dict) -> str:
    """Map a scraped name onto the bios CSV's exact spelling when possible.

    resolve_join in src/features.py matches names EXACTLY, so "Borislav Nikolic"
    in the bios and "Borislav Nikolic" with a diacritic from Wikipedia are two
    different fighters as far as the pipeline is concerned, and the bout gets
    dropped. Folding accents and case to find the existing spelling recovers
    those without touching the core matcher, which the existing 8,400 fights
    depend on.
    """
    return canon.get(fold(name), name)


def build_rows(events, columns, canon=None, verbose=True):
    """Turn parsed bouts into CSV rows matching the existing schema."""
    canon = canon or {}
    rows, unmatched_names = [], set()
    for when, name in events:
        date_iso = when.date().isoformat()
        try:
            bouts = parse_event(name)
        except Exception as exc:                        # noqa: BLE001
            print(f"  !! {name}: fetch/parse failed ({exc}) -- skipped")
            continue
        if not bouts:
            print(f"  !! {name}: no results table found -- skipped "
                  "(event may be scheduled, not completed)")
            continue
        for i, b in enumerate(bouts):
            if not b["winner"] or not b["loser"]:
                continue
            # Five rounds for title bouts and the main event (the first bout
            # listed); three otherwise. This mirrors the actual convention and
            # is the same assumption build_matchup_row's rounds= default makes.
            five = b["is_title"] or i == 0
            secs = total_seconds(b["round"], b["time"])
            row = {c: "" for c in columns}
            winner = canonicalise(b["winner"], canon)
            loser = canonicalise(b["loser"], canon)
            row.update({
                "Fight_URL": synth_fight_url(date_iso, winner, loser),
                "Fighter_1": winner,
                "Fighter_2": loser,
                "Winner": winner,
                "Weight_Class": weight_class_string(b["division"], b["is_title"]),
                "Method": b["method"],
                "End_Round": b["round"],
                "End_Time": b["time"],
                "Total_Fight_Time_Sec": secs,
                "Time_Format": ("5 Rnd (5-5-5-5-5)" if five else "3 Rnd (5-5-5)"),
                "Event_Date": date_iso,
            })
            rows.append(row)
        if verbose:
            print(f"  {date_iso}  {len(bouts):>2} bouts  {name}")
        time.sleep(POLITE_DELAY_S)
    return rows, unmatched_names


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
    ap.add_argument("--add-bio-stubs", action="store_true",
                    help="also write a fighters CSV with NaN-biometric stubs "
                         "for genuinely new fighters, so their bouts are not "
                         "dropped by the name join")
    args = ap.parse_args()

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

    print(f"local data ends {prev_max.date()}  ({len(existing)} fights)")
    events = discover_new_events(prev_max, args.limit)
    if not events:
        print("nothing new. Up to date.")
        return
    print(f"{len(events)} new event(s) to ingest"
          + (f" (limited to {args.limit})" if args.limit else "") + ":\n")

    new_rows, _ = build_rows(events, columns, canon=canon)
    print(f"\nparsed {len(new_rows)} bouts from {len(events)} event(s)")

    problems = validate(existing, new_rows, known)
    if problems:
        print("\nVALIDATION FAILED -- nothing written:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nvalidation passed: ids unique and unseen, dates after the previous "
          "max, winner is a participant, required fields present")

    stat_cols = [c for c in columns if c.startswith(("F1_", "F2_"))]
    print(f"{len(stat_cols)} per-fight stat columns left EMPTY (not zero) -- "
          "the five striking/grappling rates stay frozen for these fighters")

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
                row["Fighter_URL"] = f"wiki:fighter:{fold(n).replace(' ', '_')}"
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
