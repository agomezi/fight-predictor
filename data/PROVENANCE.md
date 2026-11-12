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

### 2026-10-08 — ESPN catch-up, 2026-08-15 -> 2026-10-03
Ingested by `scripts/refresh_data.py` (ESPN source) in two passes: an earlier
run had already produced 2026-08-22 -> 2026-09-26 in `_refreshed.csv` but was
never promoted; this pass added UFC 332 (2026-10-03, 14 bouts) and promoted
both. Fights 8,809 -> 8,900 raw; 8,749 labelled. One bio stub added (Bruce
Whitehead), biometrics blank. Fighters 4,523 -> 4,550.

Gate: all 8 test suites pass; walk-forward `+ rolling form` 0.6096 (sd 0.0239),
inside the 0.6082 +/- 0.0202 band. Previous files kept in
`backup-2026-08-15-final/` and `backup-2026-09-26/`.

Known hazard: `refresh_data.py --write` with `--out` equal to the input CSV
truncates the input before reading it (the output is opened first). Always
pass a distinct `--out`.

### 2026-11-12 — ESPN catch-up, 2026-10-10 -> 2026-11-07
Ingested by `scripts/refresh_data.py` (ESPN source), five cards: 10-10 (12),
10-17 (13), 10-24 (14, UFC 333), 10-31 (12), 11-07 (14). Fights 8,900 ->
8,965 raw; 8,749 -> 8,813 labelled (the 10-10 Kareckaite vs Gatto draw is
written `Draw/NC` and dropped by the label cleaner). 11 bio stubs added,
biometrics blank. Fighters 4,550 -> 4,561. 20 of 26 per-fight stat columns
carry values.

Gate: all 8 test suites pass; walk-forward `+ rolling form` 0.6011 (sd 0.0253),
inside the 0.6082 +/- 0.0202 band. It sits 0.0085 below the 2026-10-08 run,
which is a third of the fold sd. Previous files kept in `backup-2026-10-08/`.
Fight-week scoring of the five cards (`scripts/score_cards.py`): 0.562
(binomial sd 0.062) over 64 bouts.
