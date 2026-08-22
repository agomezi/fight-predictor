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
