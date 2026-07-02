"""
Evals from First Principles — Part 11: Regression Gates.

Everything the series built (a score, a golden set, a judge, a confidence
interval) now becomes one decision a CI pipeline can make on its own: ship,
or block. A regression gate is three cheap checks run before a deploy:

  1. THRESHOLD GATE  — pass only if the LOWER bound of the score's bootstrap
     confidence interval clears the bar. A point estimate above the line is
     not enough; the interval has to clear it too.
  2. CONTAMINATION   — an n-gram overlap check that flags an eval item that
     leaked into the training data (a leaked item inflates the score).
  3. DRIFT           — compare the score DISTRIBUTIONS of two runs, not just
     their means, so a shift is caught even when the averages look close.

We run all three on two candidate runs of the same tiny eval set: run A
(a clear pass) and run B (an ambiguous one that games the point estimate).
Pure Python + NumPy, offline, deterministic (fixed seed for the bootstrap).
"""

import numpy as np

# --- The eval set ---------------------------------------------------------
# Same running idea as the rest of the series: a small labeled set, graded.
# Each item gets a per-item quality score in {0.0, 0.5, 1.0} (0 = wrong,
# 0.5 = partial, 1.0 = correct). Two model versions each produce 20 scores
# on the SAME 20 items, so A and B are directly comparable.
RUN_A = [1.0] * 17 + [0.5] * 2 + [0.0] * 1   # strong candidate: mean 0.90
RUN_B = [1.0] * 12 + [0.5] * 5 + [0.0] * 3   # weaker candidate: mean 0.725

THRESHOLD = 0.70   # the quality bar the gate defends
ALPHA = 0.05       # 95% confidence interval
N_BOOT = 10000     # bootstrap resamples


# --- Check 1: the CI threshold gate ---------------------------------------
def bootstrap_ci(scores, n_boot, alpha, rng):
    """Resample the scores with replacement n_boot times, re-average each
    time, and read the (alpha/2, 1-alpha/2) percentiles of those means.
    Returns (point_estimate, ci_low, ci_high). This is the Part 6 idea."""
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))     # n_boot resamples of size n
    boot_means = arr[idx].mean(axis=1)             # the metric on each resample
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def threshold_gate(scores, threshold, rng):
    """PASS iff the CI's LOWER bound clears the threshold, not just the mean."""
    mean, lo, hi = bootstrap_ci(scores, N_BOOT, ALPHA, rng)
    passed = lo >= threshold
    return mean, lo, hi, passed


# --- Check 2: n-gram contamination ----------------------------------------
def word_ngrams(text, n):
    """The set of word-level n-grams (tuples) in a lowercased string."""
    tokens = text.lower().split()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def contamination_overlap(eval_text, train_ngrams, n):
    """Fraction of the eval item's n-grams that also appear in training.
    1.0 means every n-gram of the item is somewhere in the training data."""
    ev = word_ngrams(eval_text, n)
    if not ev:
        return 0.0
    hit = sum(1 for g in ev if g in train_ngrams)
    return hit / len(ev)


# --- Check 3: distribution drift (Population Stability Index) --------------
def bucketize(scores, edges):
    """Count scores into buckets defined by edges (here {0.0}, {0.5}, {1.0})."""
    counts = [0] * len(edges)
    for s in scores:
        counts[edges.index(s)] += 1
    return counts


def psi(counts_a, counts_b):
    """Population Stability Index between two runs' bucket distributions:
    sum over buckets of (pA - pB) * ln(pA / pB). Symmetric; 0 = identical.
    Rule of thumb: <0.10 stable, 0.10-0.25 moderate drift, >0.25 significant."""
    na, nb = sum(counts_a), sum(counts_b)
    total = 0.0
    for a, b in zip(counts_a, counts_b):
        pa, pb = a / na, b / nb
        total += (pa - pb) * np.log(pa / pb)
    return float(total)


if __name__ == "__main__":
    line = "=" * 72
    rng = np.random.default_rng(0)   # fixed seed -> reproducible bootstrap

    print(line)
    print("PART 11 - REGRESSION GATES: would you stake a deploy on this run?")
    print(line)

    # ---- Check 1: CI threshold gate on two candidate runs ----------------
    print(f"\nCHECK 1 - CI THRESHOLD GATE  (bar = {THRESHOLD:.2f}, 95% bootstrap CI)")
    print("PASS only if the CI LOWER bound clears the bar, not just the mean.")
    print("-" * 72)
    verdicts = {}
    for name, scores in (("A", RUN_A), ("B", RUN_B)):
        mean, lo, hi, passed = threshold_gate(scores, THRESHOLD, rng)
        verdicts[name] = passed
        tag = "PASS" if passed else "FAIL"
        print(f"  run {name}: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  "
              f"low {'>=' if passed else '< '} {THRESHOLD:.2f}  -> {tag}")
    print("  run A clears the bar with room to spare; run B's mean is above")
    print("  the bar but its CI reaches below it, so the gate refuses to ship.")

    # ---- Check 2: contamination (n-gram overlap) -------------------------
    N = 3   # word trigrams
    train_docs = [
        "the capital of france is the city of paris",
        "spiders are arachnids with eight jointed legs and two body parts",
        "shakespeare wrote many famous plays during his career",
        "what is the capital of france",   # <- an eval question pasted into training
    ]
    eval_items = [
        ("E1", "what is the capital of france"),
        ("E2", "how many legs does a spider have"),
        ("E3", "who wrote the play romeo and juliet"),
        ("E4", "what is the boiling point of water"),
    ]
    train_ngrams = set()
    for doc in train_docs:
        train_ngrams |= word_ngrams(doc, N)

    LEAK_BAR = 0.80
    print(f"\nCHECK 2 - CONTAMINATION  (word {N}-grams, flag overlap >= {LEAK_BAR:.2f})")
    print("An eval item that leaked into training data inflates the score.")
    print("-" * 72)
    for eid, text in eval_items:
        ov = contamination_overlap(text, train_ngrams, N)
        flag = "LEAKED" if ov >= LEAK_BAR else "clean"
        print(f"  {eid}: overlap={ov:.2f}  -> {flag}   \"{text}\"")
    print("  E1 is fully present in training (flagged); E4 shares only the")
    print("  common phrase 'what is the' (0.20), so it is not contamination.")

    # ---- Check 3: distribution drift -------------------------------------
    edges = [0.0, 0.5, 1.0]
    counts_a = bucketize(RUN_A, edges)
    counts_b = bucketize(RUN_B, edges)
    drift = psi(counts_a, counts_b)
    if drift > 0.25:
        drift_tag = "SIGNIFICANT drift"
    elif drift > 0.10:
        drift_tag = "moderate drift"
    else:
        drift_tag = "stable"
    print("\nCHECK 3 - DRIFT  (score distribution of run A vs run B, PSI)")
    print("Means can look close while the shape of the distribution moves.")
    print("-" * 72)
    print(f"  buckets           {edges}")
    print(f"  run A counts      {counts_a}   (n={sum(counts_a)})")
    print(f"  run B counts      {counts_b}   (n={sum(counts_b)})")
    print(f"  PSI = sum (pA-pB)*ln(pA/pB) = {drift:.3f}  -> {drift_tag}")

    # ---- The gate's verdict ----------------------------------------------
    ship = verdicts["B"] and (drift <= 0.25)
    print("\n" + line)
    print("GATE VERDICT on candidate B (the run we were about to deploy):")
    print(f"  CI threshold gate : {'PASS' if verdicts['B'] else 'FAIL'}")
    print(f"  distribution drift : {drift_tag}")
    print(f"  DECISION: {'SHIP' if ship else 'BLOCK THE DEPLOY'}")
    print("  A number, not a vibe, stops the regression from reaching prod.")
    print(line)

# Expected output:
# ========================================================================
# PART 11 - REGRESSION GATES: would you stake a deploy on this run?
# ========================================================================
#
# CHECK 1 - CI THRESHOLD GATE  (bar = 0.70, 95% bootstrap CI)
# PASS only if the CI LOWER bound clears the bar, not just the mean.
# ------------------------------------------------------------------------
#   run A: mean=0.900  95% CI=[0.775, 1.000]  low >= 0.70  -> PASS
#   run B: mean=0.725  95% CI=[0.550, 0.875]  low <  0.70  -> FAIL
#   run A clears the bar with room to spare; run B's mean is above
#   the bar but its CI reaches below it, so the gate refuses to ship.
#
# CHECK 2 - CONTAMINATION  (word 3-grams, flag overlap >= 0.80)
# An eval item that leaked into training data inflates the score.
# ------------------------------------------------------------------------
#   E1: overlap=1.00  -> LEAKED   "what is the capital of france"
#   E2: overlap=0.00  -> clean   "how many legs does a spider have"
#   E3: overlap=0.00  -> clean   "who wrote the play romeo and juliet"
#   E4: overlap=0.20  -> clean   "what is the boiling point of water"
#   E1 is fully present in training (flagged); E4 shares only the
#   common phrase 'what is the' (0.20), so it is not contamination.
#
# CHECK 3 - DRIFT  (score distribution of run A vs run B, PSI)
# Means can look close while the shape of the distribution moves.
# ------------------------------------------------------------------------
#   buckets           [0.0, 0.5, 1.0]
#   run A counts      [1, 2, 17]   (n=20)
#   run B counts      [3, 5, 12]   (n=20)
#   PSI = sum (pA-pB)*ln(pA/pB) = 0.334  -> SIGNIFICANT drift
#
# ========================================================================
# GATE VERDICT on candidate B (the run we were about to deploy):
#   CI threshold gate : FAIL
#   distribution drift : SIGNIFICANT drift
#   DECISION: BLOCK THE DEPLOY
#   A number, not a vibe, stops the regression from reaching prod.
# ========================================================================
