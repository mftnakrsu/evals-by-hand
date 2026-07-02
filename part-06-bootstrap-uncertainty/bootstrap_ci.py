"""
Evals from First Principles — Part 6: Uncertainty (Bootstrap CIs)

A single score is a point estimate. Report the interval instead. The bootstrap
turns ONE labeled set into a confidence interval: resample the set with
replacement many times, recompute the metric on each resample, and read the
2.5th and 97.5th percentiles as a 95% CI. We do it by hand on a fixed 30-item
correct/incorrect set, print a few resample draws so the mechanic is visible,
then show the interval is wide at n=30 and narrows as the set grows.
Pure Python + NumPy, offline, deterministic (fixed seed).
"""

import numpy as np

# --- The labeled set ------------------------------------------------------
# 30 graded outputs from one eval run: 1 = correct, 0 = incorrect.
# This is the SAMPLE. The true accuracy is unknown; 0.70 is only our estimate.
DATA = np.array([
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
    1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
    0, 1, 1, 0, 0, 1, 1, 0, 1, 1,
])


def bootstrap(data, B, seed, keep=0):
    """Resample `data` with replacement B times; return the metric on each.

    Each replicate draws n indices uniformly from 0..n-1 (WITH replacement),
    so some items appear twice and some not at all. The metric is the mean
    (accuracy) of the resampled labels. Returns the B replicate values and,
    if keep > 0, the first `keep` index draws so the reader can inspect them.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    reps = np.empty(B)
    draws = []
    for b in range(B):
        idx = rng.integers(0, n, size=n)   # n draws, with replacement
        if b < keep:
            draws.append(idx)
        reps[b] = data[idx].mean()
    return reps, draws


def ci95(reps):
    """Read the 2.5th and 97.5th percentiles: the 95% bootstrap interval."""
    lo = np.percentile(reps, 2.5)
    hi = np.percentile(reps, 97.5)
    return lo, hi


if __name__ == "__main__":
    line = "=" * 72
    dash = "-" * 72
    print(line)
    print("PART 6 - UNCERTAINTY: the bootstrap turns one score into an interval.")
    print(line)

    n = len(DATA)
    correct = int(DATA.sum())
    point = DATA.mean()
    print("\nThe labeled set (30 items, 1 = correct, 0 = incorrect):")
    print("  " + " ".join(str(x) for x in DATA))
    print(f"  correct: {correct} / {n}")
    print(f"  point estimate:  accuracy = {correct}/{n} = {point:.4f}")

    print("\n" + dash)
    print("THE MECHANIC: resample 30 items WITH REPLACEMENT, recompute accuracy.")
    print("Three example resamples (fixed seed, numpy default_rng(0)):")
    print(dash)
    B = 10000
    reps, draws = bootstrap(DATA, B, seed=0, keep=3)
    for i, idx in enumerate(draws, start=1):
        resampled = DATA[idx]
        c = int(resampled.sum())
        print(f"  draw {i}: indices = {idx.tolist()}")
        print(f"          correct in resample: {c} / {n}  ->  accuracy = {c / n:.4f}")
    print("  (repeated indices are the point: 'with replacement' reshuffles weight.)")

    print("\n" + dash)
    print(f"FULL BOOTSTRAP: B = {B} resamples, then read the percentiles.")
    print(dash)
    lo, hi = ci95(reps)
    print(f"  bootstrap replicates    = {B}")
    print(f"  mean of replicates      = {reps.mean():.4f}")
    print(f"  2.5th percentile        = {lo:.4f}")
    print(f"  97.5th percentile       = {hi:.4f}")
    print(f"  95% CI                  = [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
    print(f"  -> report: accuracy = {point:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")
    print("     the point estimate alone hides how much this set could wobble.")

    print("\n" + dash)
    print("WIDE AT n=30, NARROWER AS n GROWS (same 0.70 accuracy, bigger set):")
    print(dash)
    print(f"  {'n':>5}   {'95% CI':<20}  {'width':>7}")
    prev_width = None
    for k in (1, 4, 16):
        big = np.tile(DATA, k)          # same 21/30 ratio, k times as many items
        reps_k, _ = bootstrap(big, B, seed=0)
        lo_k, hi_k = ci95(reps_k)
        w = hi_k - lo_k
        ratio = "" if prev_width is None else f"  ({prev_width / w:.2f}x tighter)"
        print(f"  {len(big):>5}   [{lo_k:.4f}, {hi_k:.4f}]  {w:>7.4f}{ratio}")
        prev_width = w
    print("  -> width roughly halves each time n quadruples (the ~1/sqrt(n) rule).")
    print("     n=30 gives an interval too wide to trust a 2-point difference.")
    print(line)

# Expected output:
# ========================================================================
# PART 6 - UNCERTAINTY: the bootstrap turns one score into an interval.
# ========================================================================
#
# The labeled set (30 items, 1 = correct, 0 = incorrect):
#   1 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 0 0 1 1 0 1 1
#   correct: 21 / 30
#   point estimate:  accuracy = 21/30 = 0.7000
#
# ------------------------------------------------------------------------
# THE MECHANIC: resample 30 items WITH REPLACEMENT, recompute accuracy.
# Three example resamples (fixed seed, numpy default_rng(0)):
# ------------------------------------------------------------------------
#   draw 1: indices = [25, 19, 15, 8, 9, 1, 2, 0, 5, 24, 19, 27, 15, 18, 29, 21, 18, 16, 16, 28, 8, 24, 20, 0, 11, 25, 16, 1, 22, 21]
#           correct in resample: 21 / 30  ->  accuracy = 0.7000
#   draw 2: indices = [25, 5, 2, 25, 0, 16, 2, 8, 14, 12, 12, 0, 0, 3, 0, 20, 15, 19, 7, 18, 22, 11, 13, 29, 24, 29, 11, 20, 28, 19]
#           correct in resample: 23 / 30  ->  accuracy = 0.7667
#   draw 3: indices = [25, 20, 21, 11, 26, 4, 17, 21, 25, 15, 11, 9, 12, 14, 21, 26, 2, 28, 15, 10, 20, 17, 7, 9, 21, 17, 15, 10, 22, 11]
#           correct in resample: 25 / 30  ->  accuracy = 0.8333
#   (repeated indices are the point: 'with replacement' reshuffles weight.)
#
# ------------------------------------------------------------------------
# FULL BOOTSTRAP: B = 10000 resamples, then read the percentiles.
# ------------------------------------------------------------------------
#   bootstrap replicates    = 10000
#   mean of replicates      = 0.6998
#   2.5th percentile        = 0.5333
#   97.5th percentile       = 0.8667
#   95% CI                  = [0.5333, 0.8667]  (width 0.3333)
#   -> report: accuracy = 0.70, 95% CI [0.53, 0.87]
#      the point estimate alone hides how much this set could wobble.
#
# ------------------------------------------------------------------------
# WIDE AT n=30, NARROWER AS n GROWS (same 0.70 accuracy, bigger set):
# ------------------------------------------------------------------------
#       n   95% CI                  width
#      30   [0.5333, 0.8667]   0.3333
#     120   [0.6167, 0.7833]   0.1667  (2.00x tighter)
#     480   [0.6583, 0.7417]   0.0833  (2.00x tighter)
#   -> width roughly halves each time n quadruples (the ~1/sqrt(n) rule).
#      n=30 gives an interval too wide to trust a 2-point difference.
# ========================================================================
