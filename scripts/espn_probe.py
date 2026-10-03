"""Dump ONE event's per-fight stats from ESPN, to eyeball against Fightcenter.

WHY THIS EXISTS AND WHY IT ONLY READS. The five stat columns -- sig_landed_pm,
sig_absorbed_pm, td_landed_p15m, sub_att_p15m, ctrl_frac -- have been frozen
since ufcstats.com started serving a JavaScript proof-of-work interstitial.
Wikipedia carries results but no stats, so the weekly discovery job refreshes 9
of the 14 rolling features and leaves 5 alone.

ESPN's core API serves those five, per fight, as JSON, to a plain client:

    https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events

Verified against the Fightcenter UI for UFC 331 (Van vs. Pantoja): KD, total
strikes, significant strikes, head/body/leg, control time, takedowns and
submission attempts all match the rendered page exactly.

WHAT THIS IS NOT. ufc.com/athlete pages also publish four of the five, but as
CAREER AVERAGES AS OF TODAY. Joining those onto a 2019 row backfills a fighter's
future into their past -- the leakage the chronological split exists to prevent.
This script targets the per-competitor endpoint, which is a genuine box score
for that one bout, and is therefore safe to join by date.

This probe writes ONE CSV to the scratch path given and touches nothing the
pipeline reads. Nothing here appends to the dataset; that comes later, behind
the existing pending-CSV + pull-request review flow.

    python scripts/espn_probe.py --event 600060963
    python scripts/espn_probe.py --event 600060963 --out probe.csv
    python scripts/espn_probe.py --list 2026

Then open the same card on ESPN and compare by eye:
    https://www.espn.com/mma/fightcenter/_/id/600060963/league/ufc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CORE = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc"
UA = "fight-predictor personal research (github.com/agomezi/fight-predictor)"

# ESPN answers a plain client, so this is politeness rather than evasion. One
# event is ~25 requests: the event, then two competitors per bout, each needing
# an athlete name and a statistics document.
POLITE_DELAY_S = 0.5

# CONTROL TIME IS NOT TRACKED IN THE OLD ERA, AND ESPN REPORTS THE GAP AS 0.0.
#
# Measured while probing: timeInControl comes back 0.0 for 2005, 2010 and 2014
# events and 369.0 for 2019. A real 0:00 and "nobody recorded it" are the same
# number in this feed, and they are not the same fact. Writing 0 for the second
# case is exactly the trap data/PROVENANCE.md warns about -- a blank reads as
# unknown and earns a missing flag, a zero reads as a measurement of nothing and
# silently drags ctrl_frac toward zero across a decade of rows.
#
# So control time is written EMPTY, never 0, for events before this date. The
# cutoff is deliberately a constant you can argue with rather than a magic
# number buried in a branch.
CTRL_TRACKED_FROM = "2015-01-01"

# ESPN stat name -> the CSV's column, for the values that map 1:1. The
# head/body/leg and distance/clinch/ground columns are sums over three position
# buckets and are handled separately in extract_stats.
DIRECT_STAT_MAP = {
    "KD": "knockDowns",
    "Sig_Landed": "sigStrikesLanded",
    "Sig_Att": "sigStrikesAttempted",
    "TD_Landed": "takedownsLanded",
    "TD_Att": "takedownsAttempted",
    "Sub_Att": "submissions",
}

# Each of these CSV columns is the sum of the same strike type across the three
# positions ESPN splits by. Verified: head + body + leg == sigStrikesLanded.
TARGET_PARTS = {
    "Head": ("sigDistanceHeadStrikesLanded", "sigClinchHeadStrikesLanded",
             "sigGroundHeadStrikesLanded"),
    "Body": ("sigDistanceBodyStrikesLanded", "sigClinchBodyStrikesLanded",
             "sigGroundBodyStrikesLanded"),
    "Leg": ("sigDistanceLegStrikesLanded", "sigClinchLegStrikesLanded",
            "sigGroundLegStrikesLanded"),
}
POSITION_PARTS = {
    "Distance": ("sigDistanceHeadStrikesLanded", "sigDistanceBodyStrikesLanded",
                 "sigDistanceLegStrikesLanded"),
    "Clinch": ("sigClinchHeadStrikesLanded", "sigClinchBodyStrikesLanded",
               "sigClinchLegStrikesLanded"),
    "Ground": ("sigGroundHeadStrikesLanded", "sigGroundBodyStrikesLanded",
               "sigGroundLegStrikesLanded"),
}

# The columns this probe can fill, in the dataset's own order. Deliberately a
# subset of SCHEMA_COLUMNS in refresh_data.py: this script proves the stats can
# be read, it does not write training data.
PROBE_COLUMNS = (
    "Event_Date", "Event_Name", "Card_Segment", "Bout_Order",
    "Fighter_1", "Fighter_2", "Winner", "Weight_Class", "Time_Format",
    "F1_KD", "F2_KD", "F1_Sig_Landed", "F1_Sig_Att", "F2_Sig_Landed",
    "F2_Sig_Att", "F1_TD_Landed", "F2_TD_Landed", "F1_TD_Att", "F2_TD_Att",
    "F1_Sub_Att", "F2_Sub_Att", "F1_Ctrl_Sec", "F2_Ctrl_Sec",
    "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg",
    "F1_Distance", "F2_Distance", "F1_Clinch", "F2_Clinch",
    "F1_Ground", "F2_Ground",
)


def get_json(url: str, retries: int = 3) -> dict:
    """One GET returning parsed JSON, with a short backoff on transient failure."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def list_events(year: int) -> list:
    """(id, date, name) for every UFC event ESPN has in a calendar year."""
    doc = get_json(f"{CORE}/events?limit=400&dates={year}")
    out = []
    for item in doc.get("items", []):
        event = get_json(item["$ref"])
        out.append((event["id"], event.get("date", "")[:10], event.get("name", "")))
        time.sleep(POLITE_DELAY_S)
    return sorted(out, key=lambda r: r[1])


def parse_control_seconds(display_value):
    """ESPN's control time is a "M:SS" display string; the CSV wants seconds.

    Returns None when the value is absent or unparseable, so the caller can
    write a blank rather than inventing a zero.
    """
    if not display_value or ":" not in str(display_value):
        return None
    try:
        minutes, seconds = str(display_value).split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


def extract_stats(stats_doc: dict) -> dict:
    """Flatten one competitor's statistics document into CSV-shaped values.

    Returns a dict keyed WITHOUT the F1_/F2_ prefix -- the caller adds it once
    it knows which corner this fighter is. Control time is returned as
    "ctrl_display" (the raw "M:SS") so the era rule can be applied by the
    caller, which is the only place that knows the event date.
    """
    flat = {}
    for category in stats_doc.get("splits", {}).get("categories", []):
        for stat in category.get("stats", []):
            flat[stat.get("name")] = stat

    def value(name):
        entry = flat.get(name)
        return None if entry is None else entry.get("value")

    out = {}
    for column, espn_name in DIRECT_STAT_MAP.items():
        raw = value(espn_name)
        out[column] = "" if raw is None else int(raw)

    for column, parts in {**TARGET_PARTS, **POSITION_PARTS}.items():
        values = [value(p) for p in parts]
        out[column] = "" if all(v is None for v in values) else int(
            sum(v or 0 for v in values))

    entry = flat.get("timeInControl") or {}
    out["ctrl_display"] = entry.get("displayValue")
    return out


def fetch_event(event_id: str, verbose: bool = True) -> tuple:
    """Fetch one event and return (rows, event_name, event_date, problems).

    Each row is a dict over PROBE_COLUMNS. `problems` collects anything that
    looked wrong but did not justify aborting -- reported at the end rather
    than left to scroll past.
    """
    event = get_json(f"{CORE}/events/{event_id}?lang=en&region=us")
    event_name = event.get("name", "")
    event_date = event.get("date", "")[:10]
    ctrl_tracked = event_date >= CTRL_TRACKED_FROM

    if verbose:
        print(f"{event_name}   {event_date}")
        print(f"{len(event.get('competitions', []))} bout(s)")
        if not ctrl_tracked:
            print(f"  control time NOT tracked before {CTRL_TRACKED_FROM} "
                  "-- F1/F2_Ctrl_Sec will be written EMPTY, not 0")
        print()

    rows, problems = [], []
    for order, competition in enumerate(event.get("competitions", [])):
        segment = (competition.get("cardSegment") or {}).get("description", "")

        # NOTE ON ORDERING. competitions[] is the list of BOUTS on the card, and
        # it is NOT sorted by card position -- on UFC 331 competitions[0] is an
        # early prelim. So the "first bout is the main event, therefore five
        # rounds" heuristic that refresh_data.py uses for Wikipedia would be
        # wrong here. It is also unnecessary: `description` states the round
        # format outright, so this reads it instead of inferring it.
        time_format = competition.get("description", "")

        corners = []
        for competitor in competition.get("competitors", []):
            athlete = get_json(competitor["athlete"]["$ref"])
            time.sleep(POLITE_DELAY_S)
            name = athlete.get("displayName", "")

            stats = {}
            stats_ref = (competitor.get("statistics") or {}).get("$ref")
            if stats_ref:
                try:
                    stats = extract_stats(get_json(stats_ref))
                    time.sleep(POLITE_DELAY_S)
                except RuntimeError as exc:
                    problems.append(f"{name}: statistics fetch failed ({exc})")
            else:
                problems.append(f"{name}: no statistics document on this bout")

            corners.append({
                "name": name,
                "winner": bool(competitor.get("winner")),
                "order": competitor.get("order", 0),
                "stats": stats,
            })

        if len(corners) != 2:
            problems.append(
                f"bout {competition.get('id')} has {len(corners)} competitor(s), "
                "expected 2 -- skipped")
            continue

        corners.sort(key=lambda c: c["order"])
        first, second = corners

        winners = [c for c in corners if c["winner"]]
        if len(winners) == 1:
            winner_name = winners[0]["name"]
        else:
            # A draw or no-contest has no single winner, and so does a bout ESPN
            # has not finished grading. Left blank rather than guessed; the real
            # ingest path must decide what to do with it, loudly.
            winner_name = ""
            problems.append(
                f"{first['name']} vs {second['name']}: {len(winners)} winner(s) "
                "flagged -- draw, no-contest, or not yet final")

        row = {column: "" for column in PROBE_COLUMNS}
        row.update({
            "Event_Date": event_date,
            "Event_Name": event_name,
            "Card_Segment": segment,
            "Bout_Order": order,
            "Fighter_1": first["name"],
            "Fighter_2": second["name"],
            "Winner": winner_name,
            "Time_Format": time_format,
        })

        for prefix, corner in (("F1", first), ("F2", second)):
            stats = corner["stats"]
            for column in DIRECT_STAT_MAP:
                row[f"{prefix}_{column}"] = stats.get(column, "")
            for column in {**TARGET_PARTS, **POSITION_PARTS}:
                row[f"{prefix}_{column}"] = stats.get(column, "")

            seconds = parse_control_seconds(stats.get("ctrl_display"))
            # Blank, never zero, when the era did not record it -- see
            # CTRL_TRACKED_FROM above.
            row[f"{prefix}_Ctrl_Sec"] = (
                "" if not ctrl_tracked or seconds is None else seconds)

        rows.append(row)
        if verbose:
            mark = "*" if winner_name == first["name"] else " "
            print(f"  [{segment or '?':<12}] {time_format:<18} "
                  f"{first['name']}{mark} vs {second['name']}")

    return rows, event_name, event_date, problems


def reconcile(rows: list) -> list:
    """Cross-checks that must hold if the mapping onto the schema is right.

    These are the checks that caught the ordering assumption and the control
    time zero. Reported, never silently corrected -- a probe that quietly fixes
    its own input teaches you nothing about the source.
    """
    problems = []
    for row in rows:
        label = f"{row['Fighter_1']} vs {row['Fighter_2']}"
        for prefix in ("F1", "F2"):
            target = [row[f"{prefix}_{c}"] for c in ("Head", "Body", "Leg")]
            position = [row[f"{prefix}_{c}"] for c in
                        ("Distance", "Clinch", "Ground")]
            landed = row[f"{prefix}_Sig_Landed"]
            if "" in target or landed == "":
                continue
            if sum(target) != landed:
                problems.append(
                    f"{label} [{prefix}]: head+body+leg={sum(target)} but "
                    f"Sig_Landed={landed}")
            if "" not in position and sum(position) != landed:
                problems.append(
                    f"{label} [{prefix}]: distance+clinch+ground={sum(position)} "
                    f"but Sig_Landed={landed}")
            if row[f"{prefix}_Sig_Att"] != "" and landed > row[f"{prefix}_Sig_Att"]:
                problems.append(f"{label} [{prefix}]: landed > attempted")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", help="ESPN event id, e.g. 600060963")
    parser.add_argument("--list", type=int, metavar="YEAR",
                        help="list event ids for a year instead of probing one")
    parser.add_argument("--out", default=None,
                        help="write the rows to this CSV (default: print only)")
    args = parser.parse_args()

    if args.list:
        for event_id, date, name in list_events(args.list):
            print(f"  {event_id}  {date}  {name}")
        return

    if not args.event:
        parser.error("pass --event <id>, or --list <year> to find one")

    rows, name, date, problems = fetch_event(args.event)
    print(f"\nparsed {len(rows)} bout(s) from {name}")

    problems += reconcile(rows)
    if problems:
        print(f"\n  {len(problems)} thing(s) to look at:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("reconciliation passed: head+body+leg and "
              "distance+clinch+ground both sum to Sig_Landed, landed <= attempted")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PROBE_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}  ({len(rows)} rows)")

    print(f"\nCompare by eye against the same card:\n"
          f"  https://www.espn.com/mma/fightcenter/_/id/{args.event}/league/ufc")


if __name__ == "__main__":
    main()
