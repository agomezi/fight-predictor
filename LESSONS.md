# Methodology notes

Working notes on what this project taught, kept separate from the README because
the README is about the model and this is about the measuring. Almost none of it
is UFC-specific, and it is the part that transfers.

Not a summary of results — those are in the README. This is the list of things
that were wrong, how they were caught, and what the general form of each mistake
is.

---

## 1. Build the ruler before the thing you measure

The evaluation harness was built after the first model and before the features.
That ordering was luck rather than design, and it is the single decision the rest
of the project depended on.

The concrete payoff: twelve feature and model ideas were measured and eleven were
retired. Without a harness that could tell a real gain from a moved noise floor,
several of them would have shipped on the strength of a clean-looking single
number, and the project would now be a model with eleven unfalsifiable
improvements in it.

**General form:** if you cannot measure an improvement, you cannot have one. Time
spent on the instrument is not overhead.

## 2. A single held-out split will lie to you, in both directions

Column pruning had the **best single-tail log loss of any variant measured** and
was **worse than the incumbent on the fold mean**. Live Elo looked plausible on
one tail and lost six of eight folds.

One split gives one number, and that number is a property of which rows landed in
the tail as much as of the model. Walk-forward across eight expanding windows
gives eight numbers and a spread, and the spread is what makes the mean
interpretable.

**General form:** report a distribution, not a point. A point estimate with no
error bar is a rumour.

## 3. Compare like against like — the paired bootstrap

Two models scored on one test set share their errors: a tail full of upsets makes
both look bad. Comparing two independent confidence intervals throws that shared
component away and is far too conservative — overlapping intervals routinely hide
a real difference.

Resampling the row indices **once** and scoring both models on the same resample
cancels the shared noise. Measured here, the interval on the difference came out
**13–15× tighter** than a single-model interval.

**General form:** when comparing two things measured on the same sample, pair the
comparison. The variance you care about is the variance of the difference.

## 4. Know your minimum detectable effect before you go looking

This was done far too late, and it reframed everything.

The shipping rule said a gain must "survive the fold-to-fold sd" — 0.029. But
fold-to-fold variation in *accuracy* is mostly era difficulty, which both
variants share and which cancels in the per-fold *difference*. The sd of the
differences is 0.0113, so the real minimum detectable effect is:

```
MDE = 2.80 × 0.0113 / sqrt(8) = 0.0112
```

**The rule was 2.6× stricter than the data required.** Comparing a mean against
the standard deviation of the levels rather than of the differences is a category
error, not a conservative choice.

Two further findings worth carrying:

- **More folds made it worse.** 12 folds → 0.0157, 16 → 0.0137. Each fold's test
  window shrinks faster than `sqrt(k)` helps, and past a point measured accuracy
  degrades for reasons unrelated to anything under test.
- **The binding noise was the model seed, not the folds.** Six seeds on identical
  data spanned 0.6010–0.6107 — a range the same size as the MDE. No amount of
  fold-widening addresses that; only averaging over seeds does.

**General form:** compute the smallest effect your test can resolve *before*
running experiments. Otherwise a null result is ambiguous between "the idea is
bad" and "the test is too small", and you cannot tell which you have.

## 5. Report the seed you got, not the seed you liked

Seed 42 was the published headline at 0.6107. It was the **best of six seeds
tried**; the seed-averaged figure is 0.6059.

Nobody chose the flattering seed. It was the default, it was there first, and it
was never questioned — which is how this happens in practice rather than through
anything deliberate.

**General form:** any number that moves when you change a default needs the
default disclosed or averaged out.

## 6. Importance is not marginal value

`elo_diff` ranked **#2 of 22 features by gain-weighted importance** and adding it
changed nothing. It correlated 0.84 with `win_rate_diff`, which was itself the
better predictor of the label. The tree used Elo constantly; removing it cost
nothing, because the information was already present.

**General form:** importance measures how often a model *uses* a feature.
Ablation measures what its absence *costs*. Only the second is what you want, and
they come apart precisely when features are correlated — which is always.

## 7. A leak looks exactly like success

The reason this domain punishes carelessness: every leak makes the number go
**up**. There is no failing test, no exception, no anomaly — just a better result
than you had yesterday, which is what you were hoping for.

Three distinct leaks had to be defended against, and they are genuinely different
mistakes:

1. **Career averages.** Scraped-today aggregates joined onto a 2012 fight tell
   the model the fighter went on to never lose.
2. **Train/test contamination.** Random splits let the model train on fights
   later than the ones it is tested on. The split must be chronological, and the
   cut must land on a *date* boundary — a row-index cut splits a single event's
   card across both sides.
3. **Within-training contamination.** The subtle one. A career-to-date rate
   describes each fight using its own future. Every rolling feature must be
   computed strictly before the bout date — `<`, never `<=`, because a row
   sharing the date is either the fight itself or same-card information nobody
   had beforehand.

Strength of schedule added a **fourth shape**: a *double* boundary. The fighter's
own fights must be filtered before the cutoff, *and* each opponent must be scored
by their record before the date they were met, not by their record today. Getting
the inner one wrong leaks an opponent's later career backwards, and it does not
look like an error because the outer filter still reads correctly.

**General form:** write the test that would fail if you were cheating, before you
write the feature. Two independent angles, because they catch different mistakes:
fabricate a future and require the past to be unchanged; and shuffle the labels
and require the model to learn nothing.

## 8. Bundles hide their members — and so does reading noise as signal

Weight was measured as four columns at once, came out negative, and was recorded
as "does not pay". Split apart, one column looked free. Testing members
individually is the right lesson.

But the follow-up matters more: that "free" reading was **+0.0003 against a fold
sd of 0.026**. It was not a win either. Two errors in opposite directions, and
only re-measuring caught the second.

**General form:** a group failing does not mean each member failed. And a member
succeeding by less than your noise floor has not succeeded.

## 9. One code path, or the numbers are fiction

Training and serving must build the feature row through the same function. If
they diverge — a column order, an imputation rule, a different reference date for
age — the served predictions are wrong while every test-set metric stays clean.
Nothing goes red.

Two structural defences beat discipline here: **one shared builder**, and a test
that rebuilds every historical row through the *serving* path and requires zero
delta.

The same logic applies to a model artefact: the feature column order ships
**inside** the bundle, because a model served columns in a different order returns
confident nonsense rather than an error.

**General form:** where a mistake would be silent, make it structurally
impossible rather than a thing to remember.

## 10. NaN is not zero

Blank biometrics arrived as `0.0` in the source data, and `0.0` significant
strikes per minute is a real, terrible measurement rather than an absence. Every
missing value carries an explicit flag and the diff is imputed to a neutral zero,
so the model can tell "no advantage" from "unknown".

The same bug had a second form: a per-minute rate whose numerator skipped fights
lacking a stat while its denominator summed *all* fights, understating the rate by
~40% for exactly the fighters a data refresh was meant to help. It raised nothing.

**General form:** absence and zero are different values. Any code that conflates
them fails quietly and in the direction of looking reasonable.

## 11. The plateau is a finding, not a failure

Accuracy has not moved since the rolling-form layer landed. Eleven subsequent
ideas were measured and retired. The market ceiling is ~0.66 and this sits at
~0.61.

The temptation is to keep adding features until something clears the bar by
chance. The alternative is to say where the plateau is and why — which is both
more useful and more defensible, and is the reason the retired table is the most
interesting thing in the README.

**General form:** "we measured it and it did not work" is a result. Eleven of
them is a methodology.

---

## The one that generalises furthest

Every mistake above shares a shape: **something looked better than it was, and
the check that would have caught it did not exist yet.** Not one was found by
staring at code. They were found by building an instrument, pointing it at a
claim, and being willing to have the answer come back "no".
