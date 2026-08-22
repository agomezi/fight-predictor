"""Per-fighter fight history, and rolling features computed as of a date.

This is where the project's credibility lives. Everything in v1 was static
biometrics -- reach, height, age, stance -- because those are the only things
that cannot leak. The moment a feature summarises a fighter's PAST PERFORMANCE,
it can accidentally summarise their future too, and the model starts scoring
well by cheating.

The leak this module exists to prevent is subtle, and it is NOT the one the
chronological split handles. The split stops the test set from informing the
training set. This is a leak WITHIN training: if a fighter's win rate is
computed over their whole career, then their 2005 fight is described by a
feature that already knows how their 2015 fights turned out. Every row is
contaminated by its own future, the model learns "high career win rate implies
win", and the reported accuracy is fiction. The fighters CSV's `Wins`/`Losses`
and `SLpM` columns are exactly this, scraped in 2026, which is why
FEATURE_NAMES has never contained them.

The fix is per-row: for a fight on date D, a fighter is described only by fights
STRICTLY BEFORE D. That is what features_as_of enforces, and what
scripts/test_leakage.py proves.

Shape of the module:

    build_event_log()   one row per (fighter, fight) -- the long reshape
    HistoryIndex        that log grouped per fighter, for cheap lookup
    division_priors()   league-average rates, for shrinking thin histories
    features_as_of()    the rolling features
    elo_update()        one rating exchange
    run_elo()           the sweep that calls it in date order
    EloIndex            as-of rating lookup: for_fight() trains, as_of() serves
    shrink()            pull a rate toward a prior
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loading import load_fights
from src.features import build_feature_table

# Per-fight stat columns, named without the F1_/F2_ prefix. The raw table
# carries both corners for each of these; the reshape below turns them into
# "own" and "opponent" columns from one fighter's point of view.
#
# Declared as a wish list rather than a requirement: build_event_log keeps
# whichever are actually present and reports the rest. The scraped schema has
# drifted before, and a missing column should degrade the feature set, not crash
# the pipeline.
STAT_COLUMNS = (
    "KD",
    "Sig_Landed",
    "Sig_Att",
    "TD_Landed",
    "TD_Att",
    "Sub_Att",
    "Ctrl_Sec",
)

# Methods that count as a finish. Everything else is a decision.
FINISH_METHODS = ("KO/TKO", "Submission", "TKO - Doctor's Stoppage")

# Keys features_as_of must return, so matchup.py and the tests agree on the
# contract without importing each other's internals.
AS_OF_FEATURES = (
    "n_fights",           # prior UFC fights -- also the confidence in the rest
    "win_rate",           # shrunk toward the division prior
    "win_rate_raw",       # unshrunk, for comparison and debugging
    "last3_win_rate",     # recent form
    "finish_rate",        # of their wins, the fraction that were finishes
    "finished_rate",      # of their losses, the fraction that were finishes
    "sig_landed_pm",      # significant strikes landed per minute
    "sig_absorbed_pm",    # and absorbed -- offence and defence are different
    "td_landed_p15m",     # takedowns landed per 15 minutes (three rounds)
    "sub_att_p15m",
    "ctrl_frac",          # share of fight time in control
    "days_since_last",    # layoff. Ring rust is cheap to measure and real
    "total_fight_secs",   # cage experience, which n_fights alone misses
    "elo",                # pre-fight rating from run_elo
)

ELO_INITIAL = 1500.0
ELO_K = 24.0
# Fewer than this many prior fights and the rolling rates are mostly noise.
# Reported so a caller can qualify a prediction rather than silently trust it.
THIN_HISTORY_FIGHTS = 3

# Pseudo-count strength for shrink(). See the shrink docstring: it belongs in
# the same category as max_depth -- a hyperparameter, not a constant of nature.
SHRINK_ALPHA = 5.0

# Seconds in a 15-minute (three-round) fight, the reference window for the
# per-15-minutes rates.
SECS_PER_15M = 15.0 * 60.0


# ---------------------------------------------------------------------------
# 1. The long reshape
# ---------------------------------------------------------------------------
def build_event_log(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """One row per (fighter, fight), sorted by fighter then date.

    The feature table has one row per fight with A/B sides; rolling stats need
    one row per fighter per fight. Each fight contributes two rows, and the
    per-fight stats are joined back from the raw fights table on `Fight_URL`,
    which is in the feature table for exactly this purpose.

    Mapping a side to its stat columns goes through `a_is_f1`: fighter A's
    stats are the F1_ columns when a_is_f1 is True and the F2_ columns
    otherwise. Getting that backwards would silently swap every fighter's
    offence with their opponent's, so it is asserted below.

    Returns:
        (log, info) where log has columns:
            fighter_url, opponent_url, Fight_URL, Event_Date, Weight_Class,
            won, is_finish, Method, End_Round, fight_secs,
            own_<stat> and opp_<stat> for each available STAT_COLUMNS entry
        and info reports which stat columns were found and which were missing.
    """
    features, _ = build_feature_table(seed=seed)
    fights = load_fights()

    available = [c for c in STAT_COLUMNS
                 if f"F1_{c}" in fights.columns and f"F2_{c}" in fights.columns]
    missing = [c for c in STAT_COLUMNS if c not in available]

    keep = ["Fight_URL", "Total_Fight_Time_Sec"]
    keep += [f"F{i}_{c}" for c in available for i in (1, 2)]
    raw = fights[[c for c in keep if c in fights.columns]].copy()

    merged = features.merge(raw, on="Fight_URL", how="left", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError("Fight_URL join changed the row count -- duplicate ids?")

    a_is_f1 = merged["a_is_f1"].to_numpy(dtype=bool)

    def side(col_stem: str, for_a: bool) -> pd.Series:
        """F1_/F2_ column belonging to side A (or B) after the coin flip."""
        f1, f2 = merged[f"F1_{col_stem}"], merged[f"F2_{col_stem}"]
        take_f1 = a_is_f1 if for_a else ~a_is_f1
        return f1.where(take_f1, f2)

    is_finish = merged["Method"].isin(FINISH_METHODS).to_numpy()
    secs = pd.to_numeric(merged.get("Total_Fight_Time_Sec"), errors="coerce")

    rows = []
    for for_a in (True, False):
        block = pd.DataFrame({
            "fighter_url": merged["fighter_A_url"] if for_a else merged["fighter_B_url"],
            "opponent_url": merged["fighter_B_url"] if for_a else merged["fighter_A_url"],
            "Fight_URL": merged["Fight_URL"],
            "Event_Date": merged["Event_Date"],
            "Weight_Class": merged["Weight_Class"],
            # label is 1 when A won, so B's result is its complement.
            "won": merged["label"].to_numpy() if for_a
                   else 1 - merged["label"].to_numpy(),
            "is_finish": is_finish,
            "Method": merged["Method"],
            "End_Round": merged["End_Round"],
            "fight_secs": secs,
        })
        for c in available:
            block[f"own_{c}"] = pd.to_numeric(side(c, for_a), errors="coerce").to_numpy()
            block[f"opp_{c}"] = pd.to_numeric(side(c, not for_a), errors="coerce").to_numpy()
        rows.append(block)

    log = pd.concat(rows, ignore_index=True)
    log = log.sort_values(["fighter_url", "Event_Date"], kind="mergesort")
    log = log.reset_index(drop=True)

    info = {
        "n_fights": len(features),
        "n_log_rows": len(log),
        "n_fighters": int(log["fighter_url"].nunique()),
        "stats_available": available,
        "stats_missing": missing,
    }
    if len(log) != 2 * len(features):
        raise ValueError(f"expected {2 * len(features)} log rows, got {len(log)}")
    return log, info


class HistoryIndex:
    """The event log grouped per fighter, so a lookup touches only their rows.

    Why this exists: features_as_of is called once per fighter per fight --
    about 17,000 times on this dataset -- and scanning a 17,000-row frame each
    time is quadratic. Grouped up front, each fighter's slice is a few dozen
    rows at most, so the strictly-before filter inside features_as_of can be a
    plain boolean mask without any performance worry.

    Rows for a fighter are guaranteed sorted by Event_Date ascending.
    """

    def __init__(self, log: pd.DataFrame):
        self.columns = tuple(log.columns)
        self._by_fighter: dict[str, pd.DataFrame] = {
            url: grp.reset_index(drop=True)
            for url, grp in log.groupby("fighter_url", sort=False)
        }

    def __contains__(self, fighter_url: object) -> bool:
        return fighter_url in self._by_fighter

    def fighters(self):
        return self._by_fighter.keys()

    def rows_for(self, fighter_url: str) -> pd.DataFrame:
        """Every logged fight for this fighter, Event_Date ascending.

        An unknown fighter returns an EMPTY frame with the right columns rather
        than raising, so a debut is an ordinary case for the caller to handle
        (see features_as_of's contract) instead of an exception to catch.
        """
        got = self._by_fighter.get(fighter_url)
        if got is None:
            return pd.DataFrame(columns=list(self.columns))
        return got


# ---------------------------------------------------------------------------
# 2. Division priors (boilerplate)
# ---------------------------------------------------------------------------
def division_priors(log: pd.DataFrame) -> dict:
    """League-average rates per weight class, plus a global fallback.

    What a fighter with no usable history is assumed to look like. Computed
    from the whole log on purpose: a prior is a statement about the division,
    not about the fighter, so using all of it is not the same leak as letting a
    fighter's own future inform their own row. It is a mild one though -- the
    prior for a 1995 fight is computed partly from 2020 fights -- so it is kept
    to coarse, slow-moving quantities (a win rate is 0.5 by construction; a
    finish rate moves over decades but not over months).

    Returns {"by_division": {weight_class: {...}}, "global": {...}}.
    """
    def rates(frame: pd.DataFrame) -> dict:
        wins = frame[frame["won"] == 1]
        losses = frame[frame["won"] == 0]
        return {
            "win_rate": float(frame["won"].mean()) if len(frame) else 0.5,
            "finish_rate": float(wins["is_finish"].mean()) if len(wins) else 0.5,
            "finished_rate": float(losses["is_finish"].mean()) if len(losses) else 0.5,
        }

    return {
        "by_division": {str(k): rates(g) for k, g in log.groupby("Weight_Class")},
        "global": rates(log),
    }


def prior_for(priors: dict, weight_class: object) -> dict:
    """Division prior if the class is known, else the global one."""
    return priors["by_division"].get(str(weight_class), priors["global"])


# ---------------------------------------------------------------------------
# 3. Rolling features
# ---------------------------------------------------------------------------
def _debut_features(prior: dict, elo: float) -> dict:
    """The feature dict for a fighter with no prior fights.

    Rate features fall back to the division prior; everything a prior cannot
    supply is NaN, NEVER zero. 0.0 significant strikes per minute is a real,
    terrible value and the model must not read "unknown" as "the worst fighter
    alive". The existing pipeline imputes NaN plus a missing-flag; matchup.py
    turns each NaN diff into the neutral 0.0 + missing flag, matching it.
    """
    return {
        "n_fights": 0,
        "win_rate": prior["win_rate"],
        "win_rate_raw": prior["win_rate"],
        "last3_win_rate": prior["win_rate"],
        "finish_rate": prior["finish_rate"],
        "finished_rate": prior["finished_rate"],
        "sig_landed_pm": np.nan,
        "sig_absorbed_pm": np.nan,
        "td_landed_p15m": np.nan,
        "sub_att_p15m": np.nan,
        "ctrl_frac": np.nan,
        "days_since_last": np.nan,
        "total_fight_secs": np.nan,
        "elo": elo,
        "support": "none",
    }


def _sum_or_nan(frame: pd.DataFrame, column: str) -> float:
    """Sum a stat column that may be absent (dropped by the reshape) or NaN.

    Returns NaN when the column is missing entirely, so a per-minute rate built
    on it comes out NaN (unknown) rather than 0.0 (a real, terrible value).
    """
    if column not in frame.columns:
        return np.nan
    return float(np.nansum(frame[column].to_numpy(dtype=float)))


def features_as_of(index: "HistoryIndex", fighter_url: str, as_of_date,
                   weight_class=None, priors: dict = None,
                   elo_ratings: dict = None) -> dict:
    """This fighter's rolling features, from fights STRICTLY BEFORE as_of_date.

    THE ONE RULE. Every number returned must be computable by somebody standing
    at midnight the night before the fight. `index.rows_for(fighter_url)` gives
    the fighter's whole career; the first thing this function does is throw away
    everything dated on or after `as_of_date`.

    STRICTLY before, never `<=`. A row sharing the bout's date is either the
    bout itself -- which would hand the model the answer -- or another fight on
    the same card, which is information nobody had beforehand either.

    Args:
        index: HistoryIndex over the event log.
        fighter_url: the fighter's unique id (NOT their name -- names collide).
        as_of_date: the bout's Event_Date. Anything not strictly earlier is out.
        weight_class: the bout's division, used to pick the prior.
        priors: from division_priors(). Required for shrinkage.
        elo_ratings: {fighter_url: pre-fight rating} as-of this fight, or None
            to fall back to ELO_INITIAL.

    Returns:
        dict with exactly the keys in AS_OF_FEATURES, plus "support" in
        {"none", "thin", "ok"}.
    """
    if priors is None:
        raise ValueError("features_as_of needs priors from division_priors()")
    prior = prior_for(priors, weight_class)
    elo = ELO_INITIAL if elo_ratings is None else float(
        elo_ratings.get(fighter_url, ELO_INITIAL))

    rows = index.rows_for(fighter_url)
    # The one anti-leakage line: strictly earlier than the bout. Read the
    # fighter's frame ONCE and mask it, so no later re-filter can slip in `<=`.
    as_of = pd.Timestamp(as_of_date)
    past = rows[rows["Event_Date"] < as_of]

    n = len(past)
    if n == 0:
        return _debut_features(prior, elo)

    won = past["won"].to_numpy(dtype=float)
    wins = float(won.sum())
    win_rate_raw = wins / n

    # Recent form: the last three fights, unshrunk (three is already "recent"
    # enough that shrinkage would mostly erase the signal it exists to carry).
    last3 = past["won"].to_numpy(dtype=float)[-3:]
    last3_win_rate = float(last3.mean())

    # Finish rates are conditional: finish_rate over WINS, finished_rate over
    # LOSSES. shrink() handles the no-wins / no-losses fighter by falling out to
    # the prior, so no special-casing is needed here.
    win_rows = past[past["won"] == 1]
    loss_rows = past[past["won"] == 0]
    finishes = float(win_rows["is_finish"].sum()) if len(win_rows) else 0.0
    finished = float(loss_rows["is_finish"].sum()) if len(loss_rows) else 0.0
    finish_rate = shrink(finishes, len(win_rows), prior["finish_rate"], SHRINK_ALPHA)
    finished_rate = shrink(finished, len(loss_rows), prior["finished_rate"], SHRINK_ALPHA)

    # Per-time rates. Total cage time can be NaN or 0 in the raw data; guard it
    # so a division never produces inf, and leave the rates NaN when there is no
    # measured time to divide by.
    total_secs = _sum_or_nan(past, "fight_secs")
    if not np.isfinite(total_secs) or total_secs <= 0.0:
        sig_landed_pm = sig_absorbed_pm = np.nan
        td_landed_p15m = sub_att_p15m = ctrl_frac = np.nan
        total_fight_secs = np.nan if not np.isfinite(total_secs) else total_secs
    else:
        minutes = total_secs / 60.0
        windows_15m = total_secs / SECS_PER_15M
        sig_landed_pm = _sum_or_nan(past, "own_Sig_Landed") / minutes
        sig_absorbed_pm = _sum_or_nan(past, "opp_Sig_Landed") / minutes
        td_landed_p15m = _sum_or_nan(past, "own_TD_Landed") / windows_15m
        sub_att_p15m = _sum_or_nan(past, "own_Sub_Att") / windows_15m
        ctrl_frac = _sum_or_nan(past, "own_Ctrl_Sec") / total_secs
        total_fight_secs = total_secs

    last_date = pd.Timestamp(past["Event_Date"].max())
    days_since_last = float((as_of - last_date).days)

    support = "thin" if n < THIN_HISTORY_FIGHTS else "ok"

    return {
        "n_fights": n,
        "win_rate": shrink(wins, n, prior["win_rate"], SHRINK_ALPHA),
        "win_rate_raw": win_rate_raw,
        "last3_win_rate": last3_win_rate,
        "finish_rate": finish_rate,
        "finished_rate": finished_rate,
        "sig_landed_pm": sig_landed_pm,
        "sig_absorbed_pm": sig_absorbed_pm,
        "td_landed_p15m": td_landed_p15m,
        "sub_att_p15m": sub_att_p15m,
        "ctrl_frac": ctrl_frac,
        "days_since_last": days_since_last,
        "total_fight_secs": total_fight_secs,
        "elo": elo,
        "support": support,
    }


# ---------------------------------------------------------------------------
# 4. Elo
# ---------------------------------------------------------------------------
def elo_update(rating_a: float, rating_b: float, a_won: bool,
               k: float = ELO_K) -> tuple[float, float]:
    """One rating exchange after a fight. Returns (new_a, new_b).

    Expected score on the conventional 400-point logistic scale, then a
    K-weighted correction by how surprising the result was. The two expected
    scores sum to 1 and the two corrections are equal and opposite, so the
    exchange is exactly zero-sum: whatever A gains, B loses.
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_a = 1.0 if a_won else 0.0
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def run_elo(log: pd.DataFrame, k: float = ELO_K,
            initial: float = ELO_INITIAL) -> pd.DataFrame:
    """Sweep the whole history forward, recording each fighter's PRE-fight rating.

    Boilerplate around elo_update. One pass in date order, carrying a running
    dict; for each fight, record both ratings BEFORE applying the result, then
    apply it.

    Recording before applying is the same strictly-before rule as
    features_as_of, in its easiest-to-get-wrong form: if the rating written
    against a fight already contains that fight's outcome, the feature is the
    label in disguise and the model will look extraordinary. The ordering below
    is the only thing preventing that, which is why it is not left to the
    caller.

    Returns a frame of (Fight_URL, fighter_url, Event_Date, elo_pre, elo_post)
    -- one row per (fighter, fight), joinable back onto the event log.

    elo_post is what makes SERVING possible. A training row wants the rating
    going into that specific fight (elo_pre); a future bout wants each
    fighter's rating after everything they have already done, which is the
    elo_post of their most recent fight. Reconstructing that from elo_pre alone
    is not possible for a fighter's latest bout, because there is no next fight
    whose elo_pre would carry it -- so it is recorded here instead.

    Ordering within a single date is by the event log's order (fighter_url,
    then date), which is arbitrary but deterministic. It matters only for the
    1990s tournament cards where one fighter fought twice in a night; see
    EloIndex for how the two access modes differ there.
    """
    # One row per fight, taking each fight once rather than once per corner.
    per_fight = (log.sort_values("Event_Date", kind="mergesort")
                    .groupby("Fight_URL", sort=False)
                    .first()
                    .reset_index())
    per_fight = per_fight.sort_values("Event_Date", kind="mergesort")

    ratings: dict[str, float] = {}
    out = []
    for row in per_fight.itertuples(index=False):
        a, b = row.fighter_url, row.opponent_url
        ra = ratings.get(a, initial)
        rb = ratings.get(b, initial)
        new_a, new_b = elo_update(ra, rb, bool(row.won), k=k)
        # ra/rb are the PRE-fight ratings and are what a training row for this
        # fight may see. new_a/new_b already contain this fight's result and
        # must never be used as a feature FOR this fight -- only for later ones.
        out.append((row.Fight_URL, a, row.Event_Date, ra, new_a))
        out.append((row.Fight_URL, b, row.Event_Date, rb, new_b))
        ratings[a], ratings[b] = new_a, new_b

    return pd.DataFrame(
        out,
        columns=["Fight_URL", "fighter_url", "Event_Date", "elo_pre", "elo_post"],
    )


class EloIndex:
    """As-of Elo lookup, with the training and serving modes kept explicit.

    The mistake this class exists to prevent: passing a single
    {fighter_url: rating} snapshot into features_as_of. A snapshot taken at the
    end of history is correct for the most recent fight and wrong for every
    other one, so training on it teaches the model to read 2026 ratings off
    1998 fights -- and serving a different snapshot than training used is
    train/serve skew on top of that. Two named accessors, one source.

        for_fight(fight_url)  TRAINING. Each corner's rating going into that
                              exact fight. This is elo_pre, by construction
                              free of the fight's own result.
        as_of(date)           SERVING. Each fighter's rating after every fight
                              strictly before `date`. For a future bout that is
                              simply their current rating.

    The two agree except on one case, and it is worth knowing rather than
    smoothing over: a fighter who fought twice on the same night (the 1990s
    tournament cards). for_fight gives their second bout a rating that already
    includes the first bout -- which is correct, that result really was known
    before they walked out again. as_of(date) excludes it, because it filters
    on date alone and cannot see intra-day order. So for_fight is the more
    accurate of the two, and it is the one training uses. The disagreement is
    confined to tournament-era rows; scripts/test_history.py counts them so the
    number is known rather than assumed.
    """

    def __init__(self, log: pd.DataFrame, k: float = ELO_K,
                 initial: float = ELO_INITIAL):
        self.initial = float(initial)
        self.table = run_elo(log, k=k, initial=initial)
        self._by_fight: dict[str, dict] = {}
        for row in self.table.itertuples(index=False):
            self._by_fight.setdefault(row.Fight_URL, {})[row.fighter_url] = float(
                row.elo_pre
            )
        # Date-sorted once, so as_of is a filter rather than a re-sort.
        self._sorted = self.table.sort_values("Event_Date", kind="mergesort")

    def for_fight(self, fight_url: str) -> dict:
        """Pre-fight ratings for both corners of a known fight. Training path."""
        return self._by_fight.get(fight_url, {})

    def as_of(self, date) -> dict:
        """Every fighter's rating after all fights strictly before `date`.

        O(n) in the size of the history, so this is a serving call -- once or
        twice per prediction. Building a training matrix with it would be
        quadratic; use for_fight for that.
        """
        past = self._sorted[self._sorted["Event_Date"] < pd.Timestamp(date)]
        if past.empty:
            return {}
        last = past.groupby("fighter_url")["elo_post"].last()
        return {str(k): float(v) for k, v in last.items()}

    def latest(self) -> dict:
        """Current ratings, after the entire history."""
        if self.table.empty:
            return {}
        last = (self._sorted.groupby("fighter_url")["elo_post"].last())
        return {str(k): float(v) for k, v in last.items()}


# ---------------------------------------------------------------------------
# 5. Shrinkage
# ---------------------------------------------------------------------------
def shrink(successes: float, n: int, prior: float, alpha: float = SHRINK_ALPHA) -> float:
    """Pull an observed rate toward a prior, weighted by how much data backs it.

    Treat the prior as `alpha` pseudo-observations already in the ledger, then
    take the pooled rate:

        (successes + alpha * prior) / (n + alpha)

    With no data (n == 0) the successes term is 0 and the denominator is alpha,
    so it collapses to exactly `prior`. With a lot of data alpha is swamped and
    the answer approaches successes / n. In between it interpolates smoothly --
    no threshold, no small-n special case -- and always stays between the prior
    and the observed rate.
    """
    return (successes + alpha * prior) / (n + alpha)
