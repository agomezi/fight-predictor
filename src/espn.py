"""ESPN as the data source for completed UFC events, results AND per-fight stats.

WHY THIS REPLACED WIKIPEDIA. Wikipedia event pages carry
    Weight class | Winner | def. | Loser | Method | Round | Time | Notes
and nothing else, so 9 of the 14 rolling features refreshed and 5 -- the ones
built from sig_landed_pm, sig_absorbed_pm, td_landed_p15m, sub_att_p15m and
ctrl_frac -- stayed frozen at their last-ingest values for any fighter who had
fought since. ufcstats.com publishes those five but now serves a JavaScript
proof-of-work interstitial, and getting past it would be circumventing bot
detection, so it is not an option.

ESPN publishes both, as JSON, to a plain client. One request per event:

    https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter/{id}

carries the card segments, each bout's result, round, clock, weight class, the
fighters, and each corner's box score. Verified field-by-field against the
rendered Fightcenter page for UFC 331 (Van vs. Pantoja 2): knockdowns, total and
significant strikes, head/body/leg, control time, takedowns and submission
attempts all match.

WHAT THIS IS NOT. ufc.com/athlete publishes four of the five as CAREER AVERAGES
AS OF TODAY. Joining those onto a 2019 row backfills a fighter's future into
their past, which is the leakage the chronological split exists to prevent. The
endpoint used here is a box score for one bout and is safe to join by date.

THREE THINGS MEASURED WHILE BUILDING THIS, each of which would have been a
silent bug:

  1. Control time is not tracked before ~2015 and ESPN reports the gap as 0.0,
     not null. See CTRL_TRACKED_FROM.
  2. The core API's competitions[] is NOT ordered by card position -- on UFC 331
     competitions[0] is an early prelim. The "first bout is the main event,
     therefore five rounds" heuristic the Wikipedia path used would be wrong
     here, and is unnecessary: the round format is stated outright.
  3. Method strings already match the dataset's vocabulary exactly
     ("Decision - Unanimous", "KO/TKO", "Submission", "Decision - Split"), so
     the old METHOD_MAP regex table is not needed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

CORE = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc"
FIGHTCENTER = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter"
UA = "fight-predictor personal research (github.com/agomezi/fight-predictor)"

# ESPN answers a plain client, so this is politeness rather than evasion. The
# fightcenter endpoint is one request per event, which is why the weekly job is
# cheap: a dozen events is a dozen requests, not the ~25 per event the
# per-competitor core endpoints would cost.
POLITE_DELAY_S = 0.5

# CONTROL TIME IS NOT TRACKED IN THE OLD ERA, AND ESPN REPORTS THE GAP AS 0.0.
#
# Measured: timeInControl comes back "0:00" for 2005, 2010 and 2014 events and
# real values from 2019. A genuine 0:00 and "nobody recorded it" are the same
# number in this feed and they are not the same fact. Writing 0 for the second
# case is the trap data/PROVENANCE.md warns about -- a blank reads as unknown
# and earns a missing flag, a zero reads as a measurement of nothing and drags
# ctrl_frac toward zero across a decade of rows.
#
# So control time is written EMPTY, never 0, for events before this date. It is
# a constant you can argue with rather than a magic number inside a branch.
CTRL_TRACKED_FROM = "2015-01-01"

# ESPN's inline stat name -> (landed column, attempted column). A value like
# "147/284" carries both; `submissions` and `knockDowns` are scalars.
SPLIT_STATS = {
    "sigStrikes": ("Sig_Landed", "Sig_Att"),
    "takedowns": ("TD_Landed", "TD_Att"),
    "headStrikes": ("Head", None),
    "bodyStrikes": ("Body", None),
    "legStrikes": ("Leg", None),
}
SCALAR_STATS = {
    "knockDowns": "KD",
    "submissions": "Sub_Att",
}

# NOT FILLED: F1/F2_Distance, _Clinch, _Ground. The fightcenter payload gives
# head/body/leg (the TARGET breakdown) but not the standing/clinch/ground
# POSITION breakdown; the core API's per-competitor endpoint does, at ~25
# requests per event instead of 2. Checked before leaving them: nothing in src/
# reads those six columns, so they feed no feature. They stay blank, which
# correctly reads as unknown rather than as a measurement of zero.


def get_json(url: str, retries: int = 3) -> dict:
    """One GET returning parsed JSON, with a short backoff on transient failure."""
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


# Events ESPN files under the UFC league that are NOT UFC bouts. Contender
# Series fights are developmental -- they are not on the UFC record and the
# existing 8,809-row dataset does not contain them, so ingesting them would be
# adding a different promotion's fights to the training data.
NON_UFC_EVENT_MARKERS = ("contender series", "road to ufc", "the ultimate fighter")


def is_ufc_event(name: str) -> bool:
    """False for cards ESPN files under UFC that are not UFC bouts.

    TUF is included in the exclusion list for the prelim/house fights; the TUF
    FINALE cards are real UFC events and are named "... Finale", which does not
    match "the ultimate fighter" alone -- checked against the existing dataset,
    which contains the finales and none of the house fights.
    """
    low = (name or "").lower()
    if "finale" in low:
        return True
    return not any(marker in low for marker in NON_UFC_EVENT_MARKERS)


def list_events(year: int) -> list:
    """(event_id, date, name) for every UFC event ESPN lists in a calendar year.

    The index gives only $refs, so this costs one request per event. Used by the
    discovery path, which then filters by date before fetching any card.
    """
    index = get_json(f"{CORE}/events?limit=400&dates={year}")
    events = []
    for item in index.get("items", []):
        event = get_json(item["$ref"])
        events.append((event["id"], event.get("date", "")[:10], event.get("name", "")))
        time.sleep(POLITE_DELAY_S)
    return sorted(events, key=lambda row: row[1])


def parse_landed_attempted(display_value):
    """ESPN writes "147/284" for landed/attempted. Returns (landed, attempted).

    Either side may be None when the feed omits the stat, so the caller can
    write a blank rather than inventing a zero.
    """
    if not display_value:
        return None, None
    text = str(display_value)
    if "/" not in text:
        try:
            return int(float(text)), None
        except ValueError:
            return None, None
    landed, _, attempted = text.partition("/")
    try:
        return int(float(landed)), int(float(attempted))
    except ValueError:
        return None, None


def parse_clock_seconds(display_value):
    """"M:SS" -> seconds. None when absent or unparseable."""
    if not display_value or ":" not in str(display_value):
        return None
    minutes, _, seconds = str(display_value).partition(":")
    try:
        return int(float(minutes)) * 60 + int(float(seconds))
    except ValueError:
        return None


def extract_corner_stats(competitor: dict) -> dict:
    """One corner's box score, keyed WITHOUT the F1_/F2_ prefix.

    Control time is returned raw under "ctrl_seconds"; the era rule belongs to
    the caller, which is the only place that knows the event date.
    """
    flat = {stat.get("name"): stat for stat in competitor.get("stats", [])}
    out = {}

    for espn_name, (landed_col, attempted_col) in SPLIT_STATS.items():
        landed, attempted = parse_landed_attempted(
            (flat.get(espn_name) or {}).get("displayValue"))
        out[landed_col] = "" if landed is None else landed
        if attempted_col:
            out[attempted_col] = "" if attempted is None else attempted

    for espn_name, column in SCALAR_STATS.items():
        value = (flat.get(espn_name) or {}).get("value")
        out[column] = "" if value is None else int(float(value))

    control = flat.get("timeInControl") or {}
    out["ctrl_seconds"] = (
        parse_clock_seconds(control.get("displayValue"))
        if control.get("displayValue") is not None
        else (None if control.get("value") is None else int(float(control["value"]))))
    return out


def weight_class_string(competition: dict) -> str:
    """Match the dataset's conventions exactly.

    Non-title: "Lightweight Bout". Title: "UFC Lightweight Title Bout". Women's
    divisions keep their prefix, which is what is_womens_bout in src/features.py
    reads. ESPN puts the division in type.text and flags a title bout in
    types[], e.g. {"text": "UFC Flyweight Title"}.
    """
    division = ((competition.get("type") or {}).get("text") or "").strip()
    if not division:
        return ""
    titles = [t.get("text", "") for t in (competition.get("types") or [])]
    if any("title" in text.lower() for text in titles):
        return f"UFC {division} Title Bout"
    return f"{division} Bout"


def iter_competitions(card: dict):
    """Yield (segment_description, competition) across all card segments.

    `cards` is a dict of segments (main, prelims1, prelims2), each holding its
    own competitions list. Iterating the dict keeps bouts grouped by segment,
    which is also the order a human reads the card in.
    """
    for segment in (card.get("cards") or {}).values():
        # cardSegment is a dict on most events but a bare string on some
        # (Contender Series cards, for one), so this reads both shapes rather
        # than assuming the common one.
        raw_segment = segment.get("cardSegment")
        if isinstance(raw_segment, dict):
            description = raw_segment.get("description", "")
        else:
            description = raw_segment or ""
        for competition in segment.get("competitions", []):
            yield description, competition


def fetch_card(event_id: str) -> dict:
    """The full fightcenter payload for one event. One request."""
    return get_json(f"{FIGHTCENTER}/{event_id}")


def fetch_round_formats(event_id: str) -> dict:
    """{competition_id: "3 Rnd (5-5-5)"} for one event, from the core API.

    WHY A SECOND REQUEST. The fightcenter payload carries everything else but
    leaves `format` null on every bout, and its `note` marks only the main
    event -- so a five-round CO-MAIN would be recorded as three rounds. The core
    event payload states the true format per bout in `description`, in the
    dataset's own vocabulary ("3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)").

    Worth one extra request per event to read the round count rather than infer
    it. Inferring it from card position is what the Wikipedia path did, and
    ESPN's competitions[] is not ordered by card position.
    """
    event = get_json(f"{CORE}/events/{event_id}?lang=en&region=us")
    return {competition.get("id"): competition.get("description", "")
            for competition in event.get("competitions", [])}
