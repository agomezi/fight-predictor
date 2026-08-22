"""Scrape event context (venue, location, attendance, card tier) into data/.

WHY THIS EXISTS. The ufcstats CSVs describe fights and fighters but say nothing
about the EVENT: where it was held, how big the card was, whether it was a
numbered pay-per-view or a Fight Night. Those are known before a bout, cannot
leak an outcome, and are not derivable from anything already in the data.

WHY WIKIPEDIA AND NOT ufcstats. ufcstats.com now serves a JavaScript
proof-of-work interstitial ("Checking your browser...", robots noindex) to
non-browser clients. Getting past that programmatically would be circumventing
bot detection, so this scraper does not target it. That also means the original
CSVs are not reproducible by re-scraping -- they predate the challenge. See
HANDBACK-5 for the full note.

Wikipedia's "List of UFC events" carries the same information in ONE page under
a licence that permits reuse, so this is a single polite request rather than 700.

PROVENANCE, recorded in the output header comment and in data/PROVENANCE.md:
    source   https://en.wikipedia.org/wiki/List_of_UFC_events
    licence  CC BY-SA 4.0 (Wikipedia text)
    fetched  written into the file at scrape time
    join key event date -> Event_Date in ufc_gold_dataset_final.csv

Run from the repo root:
    python scripts/scrape_event_context.py
"""

from __future__ import annotations

import csv
import html
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "event_context.csv"
SOURCE = "https://en.wikipedia.org/wiki/List_of_UFC_events"
UA = "fight-predictor personal research (github.com/agomezi/fight-predictor)"

# Venues meaningfully above sea level, where cardio is measurably affected.
# Domain knowledge rather than scraped data, so it is listed explicitly and can
# be argued with. Elevation in feet.
HIGH_ALTITUDE = {
    "denver": 5280, "mexico city": 7350, "salt lake city": 4226,
    "calgary": 3438, "albuquerque": 5312, "monterrey": 1768,
    "johannesburg": 5751, "bogota": 8660, "quito": 9350,
}


def strip_tags(fragment: str) -> str:
    """Cell HTML -> plain text, with Wikipedia's inline CSS noise removed."""
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    # Wikipedia injects .mw-parser-output CSS into some cells; drop from the
    # first CSS-ish marker onward rather than trying to parse it.
    text = re.split(r"\.mw-parser-output|\{|\bN/?a\b", text)[0]
    return re.sub(r"\s+", " ", text).strip(" —-–\xa0")


def card_tier(event_name: str) -> str:
    """Numbered pay-per-views are the marquee cards; Fight Nights are not.

    A card-prestige proxy that is genuinely absent from the current data. It
    correlates with fighter quality without being derived from any result, so it
    cannot leak. Overlaps somewhat with the title-bout flag, which is why it is
    measured separately rather than assumed to help.
    """
    n = event_name.lower()
    if re.search(r"^ufc\s+\d+", n):
        return "numbered"
    if "fight night" in n:
        return "fight_night"
    if re.search(r"ufc on |ufc live", n):
        return "broadcast"
    if "ultimate fighter" in n:
        return "tuf_finale"
    return "other"


def parse_attendance(raw: str):
    digits = re.sub(r"[^\d]", "", raw or "")
    return int(digits) if digits else ""


def parse_date(raw: str):
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def fetch(url: str) -> str:
    """One polite request via curl. No credentials, no session, no cookies.

    encoding is pinned to UTF-8 rather than left to the platform default.
    text=True decodes with locale.getpreferredencoding(), which is cp1252 on
    Windows -- and Wikipedia is UTF-8, so any page carrying an accented fighter
    name (Prochazka, Blachowicz, Jedrzejczyk) raised UnicodeDecodeError and the
    whole refresh died. Those are precisely the names refresh_data.canonicalise
    exists to reconcile, so the failure hit exactly where it hurt most. errors
    is "replace" so one malformed byte degrades a single character instead of
    aborting a 21-event ingest.
    """
    res = subprocess.run(
        ["curl", "-sL", "-m", "60", "-A", UA, url],
        capture_output=True, text=True, check=False,
        encoding="utf-8", errors="replace",
    )
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"fetch failed (rc={res.returncode})")
    return res.stdout


def main() -> None:
    print(f"fetching {SOURCE}")
    page = fetch(SOURCE)
    tables = re.findall(
        r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>.*?</table>', page, re.S
    )
    if not tables:
        print("error: no wikitable found; the page structure changed",
              file=sys.stderr)
        sys.exit(1)

    rows_out, seen = [], set()
    for table in tables:
        header = [strip_tags(h).lower()
                  for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)][:8]
        if not any("date" in h for h in header):
            continue
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [strip_tags(c)
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
            if len(cells) < 5:
                continue
            # Layout is: #, Event, Date, Venue, Location, Attendance, Ref
            name, date_raw, venue, location = cells[1], cells[2], cells[3], cells[4]
            iso = parse_date(date_raw)
            if not iso or not name:
                continue
            key = (iso, name)
            if key in seen:
                continue
            seen.add(key)
            city = location.split(",")[0].strip().lower()
            rows_out.append({
                "Event_Date": iso,
                "event_name": name,
                "venue": venue,
                "location": location,
                "event_country": (location.split(",")[-1].strip() or ""),
                "attendance": parse_attendance(cells[5] if len(cells) > 5 else ""),
                "card_tier": card_tier(name),
                "altitude_ft": HIGH_ALTITUDE.get(city, 0),
            })

    rows_out.sort(key=lambda r: r["Event_Date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Event_Date", "event_name", "venue", "location", "event_country",
              "attendance", "card_tier", "altitude_ft"]
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# source: {SOURCE}\n")
        fh.write("# licence: CC BY-SA 4.0 (Wikipedia)\n")
        fh.write(f"# fetched: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write("# join: Event_Date -> ufc_gold_dataset_final.csv Event_Date\n")
        csv.DictWriter(fh, fieldnames=fields).writeheader()
        csv.DictWriter(fh, fieldnames=fields).writerows(rows_out)

    print(f"wrote {OUT.relative_to(REPO)}  ({len(rows_out)} events)")
    tiers = {}
    for r in rows_out:
        tiers[r["card_tier"]] = tiers.get(r["card_tier"], 0) + 1
    print("card tiers:", dict(sorted(tiers.items(), key=lambda t: -t[1])))
    print("with altitude:", sum(1 for r in rows_out if r["altitude_ft"]))
    print("with attendance:", sum(1 for r in rows_out if r["attendance"]))


if __name__ == "__main__":
    main()
