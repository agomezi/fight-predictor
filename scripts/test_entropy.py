"""Check entropy() and information_gain() against the worked examples in
hand-worked examples. Run from the repo root:

    python scripts/test_entropy.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tree import entropy, information_gain

TOL = 1e-3
failures = []


def check(name, got, expected):
    ok = abs(got - expected) < TOL
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {got:.4f}, expected {expected:.4f}")
    if not ok:
        failures.append(name)


# --- Part 2 example: 8 toy fights, Finish (F) vs Decision (D) -------------
# | Fight | Reach Adv? | Southpaw? | Outcome  |
# |   1   |    Yes     |    Yes    | Finish   |
# |   2   |    Yes     |    No     | Finish   |
# |   3   |    Yes     |    Yes    | Finish   |
# |   4   |    Yes     |    No     | Decision |
# |   5   |    No      |    Yes    | Decision |
# |   6   |    No      |    No     | Decision |
# |   7   |    No      |    Yes    | Finish   |
# |   8   |    No      |    No     | Decision |
outcome = np.array(["F", "F", "F", "D", "D", "D", "D", "F"])
reach = np.array([1, 1, 1, 1, 0, 0, 0, 0])
southpaw = np.array([1, 0, 1, 0, 1, 0, 1, 0])

check("Root entropy (4F/4D)", entropy(outcome), 1.0)
check("Pure set entropy", entropy(np.array(["F", "F", "F"])), 0.0)
check("Empty set entropy", entropy(np.array([])), 0.0)

check("IG Reach Advantage",
      information_gain(outcome, outcome[reach == 1], outcome[reach == 0]),
      0.189)
check("IG Southpaw (zero-signal)",
      information_gain(outcome, outcome[southpaw == 1], outcome[southpaw == 0]),
      0.0)

# --- Part 3 example: 6 matchups, Fighter A wins? (Y/N) --------------------
# | # | Stance Matchup | Height Diff | A wins? |
# | 1 |   Opposite     |  A Taller   |   Y     |
# | 2 |   Opposite     |  B Taller   |   Y     |
# | 3 |   Opposite     |  Even       |   Y     |
# | 4 |   Same         |  A Taller   |   Y     |
# | 5 |   Same         |  B Taller   |   N     |
# | 6 |   Same         |  Even       |   N     |
a_wins = np.array(["Y", "Y", "Y", "Y", "N", "N"])
stance = np.array(["Opp", "Opp", "Opp", "Same", "Same", "Same"])
height = np.array(["A", "B", "E", "A", "B", "E"])

check("Base entropy (4Y/2N)", entropy(a_wins), 0.918)

# Stance is binary -> information_gain() applies directly.
check("IG Stance Matchup",
      information_gain(a_wins, a_wins[stance == "Opp"],
                       a_wins[stance == "Same"]),
      0.459)

# Height Diff is a 3-way split, so compose entropy() by hand:
# IG = H(parent) - sum over branches of (|branch|/|parent|) * H(branch)
n = len(a_wins)
weighted = sum(
    (np.sum(height == v) / n) * entropy(a_wins[height == v])
    for v in ("A", "B", "E")
)
check("IG Height Diff (3-way, via entropy())",
      entropy(a_wins) - weighted, 0.251)

# --------------------------------------------------------------------------
print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks pass — entropy and information gain match the worked examples.")
