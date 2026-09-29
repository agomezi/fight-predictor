"""Name folding and aliasing, which decide whether a fighter has a history.

WHY THIS IS A GATE AND NOT A UNIT TEST DETAIL. resolve_join in src/features.py
matches names EXACTLY. A scraped name that does not fold onto the bios' spelling
becomes a SECOND fighter with an empty record, and every rolling feature for
that bout is computed from nothing. Nothing downstream fails: the row is still
written, the model still trains, accuracy barely moves. The fighter just
quietly loses their career.

That happened in the first ESPN ingest. "Klaudia Syguła" folded to
'klaudia sygua' -- NFKD does not decompose the Polish stroked l, so the
stripping step deleted it -- and did not match the bios' "Klaudia Sygula". She
had three prior UFC bouts. The same class of bug is waiting for every stroked
letter: o-slash, d-bar, h-bar, sharp s, ae.

THE OPPOSITE ERROR IS WORSE, which is why the second half of this file exists.
A fuzzy matcher aggressive enough to catch the suffix cases would merge
"Jessie Rosas" with "Jesse Rosas Jr." -- brothers, both of whom have fought in
the UFC. Splitting one fighter in two costs that fighter's history; merging two
fighters into one corrupts BOTH records and is invisible afterwards. So folding
stays conservative and genuine exceptions go in an explicit, auditable list.

Run from the repo root (venv active):
    python scripts/test_names.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.refresh_data import NAME_ALIASES, canonicalise, fold  # noqa: E402

failures = []


def check(label, passed, detail=""):
    print(f"{'[PASS]' if passed else '[FAIL]'} {label}"
          + (f"  {detail}" if detail else ""))
    if not passed:
        failures.append(label)


print("=" * 78)
print("STROKED LETTERS MUST FOLD, NOT VANISH")
print("=" * 78)
# Each of these is a letter NFKD leaves intact because the diacritic is part of
# the glyph. Before the fix, every one of them was silently deleted.
for scraped, bios in (
    ("Klaudia Syguła", "Klaudia Sygula"),      # the one that actually bit
    ("Jan Błachowicz", "Jan Blachowicz"),
    ("Michał Oleksiejczuk", "Michal Oleksiejczuk"),
    ("Robert Ruchała", "Robert Ruchala"),
    ("Søren Bak", "Soren Bak"),                 # o with stroke
    ("Mate Sanikidzeđ", "Mate Sanikidzed"),     # d with stroke
):
    check(f"{scraped!r} folds onto {bios!r}", fold(scraped) == fold(bios),
          f"-> {fold(scraped)!r}")

print()
print("=" * 78)
print("ORDINARY ACCENTS STILL FOLD")
print("=" * 78)
for scraped, bios in (
    ("José Aldo", "Jose Aldo"),
    ("Khābib Nurmagomedov", "Khabib Nurmagomedov"),
    ("Ilia Topuria", "Ilia Topuria"),
):
    check(f"{scraped!r} folds onto {bios!r}", fold(scraped) == fold(bios))

print()
print("=" * 78)
print("DISTINCT PEOPLE MUST STAY DISTINCT")
print("=" * 78)
# The failure mode that is worse than the one above. Jessie Rosas and Jesse
# Rosas Jr. are brothers; merging them would corrupt two careers at once.
for a, b in (
    ("Jessie Rosas", "Jesse Rosas Jr."),
    ("Lance Gibson", "Lance Gibson Jr."),
    ("Kai Kamaka", "Kai Kamaka III"),
    ("Eric McConico", "Eric McConico Jr."),
):
    check(f"{a!r} does NOT fold onto {b!r}", fold(a) != fold(b))

print()
print("=" * 78)
print("HAND-VERIFIED ALIASES RESOLVE")
print("=" * 78)
# canonicalise maps onto whatever spelling the bios already use, so the test
# builds the same folded index the real path does.
canon = {}
for spelling in set(NAME_ALIASES.values()):
    canon.setdefault(fold(spelling), spelling)

for scraped, expected in NAME_ALIASES.items():
    got = canonicalise(scraped, canon)
    check(f"{scraped!r} -> {expected!r}", got == expected, f"got {got!r}")

# An alias must not fire on a name it was not written for.
check("canonicalise leaves an unknown name alone",
      canonicalise("Salahdine Parnasse", canon) == "Salahdine Parnasse")

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    print("A name that does not resolve costs that fighter their entire")
    print("history, and nothing downstream will report it.")
    sys.exit(1)
print("Name resolution holds: stroked letters fold, relatives stay separate.")
