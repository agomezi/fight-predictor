# Data provenance

## ufc_gold_dataset_final.csv / ufc_fighters_final.csv
Kaggle comprehensive UFC dataset, scraped from ufcstats.com.
MD5 verified 2026-08-22: cb75677a212d0a3fbb3369688ababcf7 (fights),
ca2b1627f83fc1f7ae3e4ef4637c9639 (bios). 8,551 fights / 4,455 fighters.

**Not reproducible by re-scraping.** ufcstats.com now serves a JavaScript
proof-of-work interstitial to non-browser clients ("Checking your browser...",
robots noindex). These files predate it. Treat them as a fixed artifact.

## event_context.csv
Source:  https://en.wikipedia.org/wiki/List_of_UFC_events
Licence: CC BY-SA 4.0 (Wikipedia text)
Fetched: 2026-08-22, one request, via scripts/scrape_event_context.py
Join:    Event_Date -> ufc_gold_dataset_final.csv Event_Date
Coverage: 724 events; joins to 87.1% of the 8,400 modelled fights. 7 dates
carry two events, so a date join is ambiguous for those and takes the first.
Fields:  venue, location, event_country, attendance, card_tier, altitude_ft.
         altitude_ft is a hand-listed table of high-elevation cities in the
         scraper, i.e. domain knowledge, not scraped data.
Status:  DELIVERED BUT UNMEASURED. No model uses it yet.

## Refresh log

### 2026-08-22 — Wikipedia catch-up, 2026-03-07 -> 2026-08-15
Ingested by `scripts/refresh_data.py` from
https://en.wikipedia.org/wiki/List_of_UFC_events and the per-event articles.
21 events, 258 bouts. Fights 8,551 -> 8,809 raw; 8,400 -> 8,658 labelled,
0 unresolved.

**Five features are FROZEN, not refreshed.** Wikipedia event tables carry
Weight class / Winner / Method / Round / Time and nothing else, so
`sig_landed_pm`, `sig_absorbed_pm`, `td_landed_p15m`, `sub_att_p15m` and
`ctrl_frac` keep their pre-March values for the ~371 fighters who have fought
since. The other 9 rolling features refresh normally. This is stale, not wrong:
`src/history._rate_basis` computes each rate over only the fights that actually
carry it, so partial availability dilutes nothing.

Per-fight stat columns for new bouts are written EMPTY, never zero.

New fighters get bio stubs with BLANK biometrics (`--add-bio-stubs`), which the
missing-flag path handles. Blank is deliberate: a zero reads as a real
measurement of nothing.

Gate result: all six suites pass; walk-forward accuracy stayed inside the
required band. See the PR for the measured numbers.
