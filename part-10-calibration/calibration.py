"""
Evals from First Principles - Part 10: Calibration.

A confidence is a promise: of the answers a model tags "80% sure," about 80%
should be right. This part keeps that promise honest. On 10 trivia answers,
each carrying a stated confidence and a correct/wrong outcome, we bin by
confidence, build the reliability curve (mean confidence vs. accuracy per bin),
then reduce the whole picture to two numbers: the Expected Calibration Error
(ECE, the weighted average gap between confidence and accuracy) and the Brier
score (mean squared error of confidence against outcome). We run it twice on
the SAME confidences: once for an over-confident model, once for a
well-calibrated one, so the metrics move while the confidences stay put.

Pure Python, offline, deterministic. No NumPy needed, no randomness, no network.
"""

# --- The task -------------------------------------------------------------
# A trivia model answers 10 questions. For each it reports how sure it is
# (CONF, a probability in [0, 1]) and we later learn whether it was actually
# right (OUTCOME, 1 = correct, 0 = wrong). CONF[i] and OUTCOME[i] are the
# same question. We keep the SAME confidences for both models below; only the
# outcomes differ, which is the whole point: calibration is about whether the
# outcomes track the confidences, not about the confidences alone.
CONF = [0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0]

# Over-confident: it says 0.82 on average but is right only 40% of the time.
OUTCOME_OVER = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

# Well-calibrated: same confidences, but the outcomes now track them.
OUTCOME_CAL = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1]

# Reliability bins: three buckets that cover the confidences we see.
# Each bin is [lo, hi); the last one includes the right edge so 1.0 lands in it.
BINS = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.0)]


def bin_index(conf):
    """Which bin does this confidence fall in? Half-open [lo, hi), last is closed."""
    for i, (lo, hi) in enumerate(BINS):
        if lo == hi:                      # the degenerate top bin, exactly 1.0
            if conf == lo:
                return i
        elif lo <= conf < hi:
            return i
    return len(BINS) - 1                  # anything at the top edge -> last bin


def bin_stats(conf, outcome):
    """Group predictions into bins; return per-bin (n, mean_conf, accuracy)."""
    n = len(BINS)
    counts = [0] * n
    conf_sum = [0.0] * n
    correct = [0] * n
    for c, o in zip(conf, outcome):
        b = bin_index(c)
        counts[b] += 1
        conf_sum[b] += c
        correct[b] += o
    rows = []
    for i in range(n):
        if counts[i] == 0:
            rows.append((0, None, None))
        else:
            rows.append((counts[i], conf_sum[i] / counts[i], correct[i] / counts[i]))
    return rows


def ece(conf, outcome):
    """Expected Calibration Error: weighted average |accuracy - confidence|."""
    total = len(conf)
    rows = bin_stats(conf, outcome)
    err = 0.0
    for count, mean_conf, acc in rows:
        if count == 0:
            continue
        err += (count / total) * abs(acc - mean_conf)
    return err


def brier(conf, outcome):
    """Brier score: mean over predictions of (confidence - outcome)^2."""
    return sum((c - o) ** 2 for c, o in zip(conf, outcome)) / len(conf)


def overall_confidence(conf):
    return sum(conf) / len(conf)


def overall_accuracy(outcome):
    return sum(outcome) / len(outcome)


def bin_label(i):
    lo, hi = BINS[i]
    if lo == hi:
        return f"[{lo:.1f}]     "
    return f"[{lo:.1f}, {hi:.1f})"


def report(conf, outcome, title):
    print(title)
    print(f"  mean confidence = {overall_confidence(conf):.2f}"
          f"   accuracy = {overall_accuracy(outcome):.2f}")
    print("  reliability curve (per confidence bin):")
    print("    bin           n   mean_conf   accuracy   gap")
    rows = bin_stats(conf, outcome)
    for i, (count, mean_conf, acc) in enumerate(rows):
        if count == 0:
            print(f"    {bin_label(i)}   0      -           -         -")
            continue
        gap = abs(acc - mean_conf)
        print(f"    {bin_label(i)}   {count}   "
              f"{mean_conf:8.3f}   {acc:8.3f}   {gap:.3f}")
    e = ece(conf, outcome)
    b = brier(conf, outcome)
    print(f"  ECE   = weighted avg gap  = {e:.3f}")
    print(f"  Brier = mean (conf-out)^2 = {b:.3f}")
    return e, b


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 10 - CALIBRATION: does a stated confidence mean what it says?")
    print(line)
    print("Same 10 confidences for both models; only the outcomes differ.")
    print(f"  confidences: {CONF}")
    print()

    e_over, b_over = report(CONF, OUTCOME_OVER,
                            "OVER-CONFIDENT model  (outcomes do NOT track confidence):")
    print()
    print("-" * 72)
    e_cal, b_cal = report(CONF, OUTCOME_CAL,
                          "WELL-CALIBRATED model  (same confidences, outcomes track them):")

    print("\n" + line)
    print("What the two numbers caught:")
    print(f"  ECE   {e_over:.3f} -> {e_cal:.3f}   "
          f"(gap between promise and reality shrinks {e_over / e_cal:.1f}x)")
    print(f"  Brier {b_over:.3f} -> {b_cal:.3f}   "
          "(lower is better; rewards being right AND appropriately unsure)")
    print("The over-confident model says 0.82 but is right 0.40 of the time:")
    print("every bin sits below the diagonal. Confidence you cannot trust is")
    print("worse than no confidence at all.")
    print(line)

# Expected output:
# ========================================================================
# PART 10 - CALIBRATION: does a stated confidence mean what it says?
# ========================================================================
# Same 10 confidences for both models; only the outcomes differ.
#   confidences: [0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0]
#
# OVER-CONFIDENT model  (outcomes do NOT track confidence):
#   mean confidence = 0.82   accuracy = 0.40
#   reliability curve (per confidence bin):
#     bin           n   mean_conf   accuracy   gap
#     [0.6, 0.8)   3      0.633      0.333   0.300
#     [0.8, 1.0)   5      0.860      0.400   0.460
#     [1.0]        2      1.000      0.500   0.500
#   ECE   = weighted avg gap  = 0.420
#   Brier = mean (conf-out)^2 = 0.432
#
# ------------------------------------------------------------------------
# WELL-CALIBRATED model  (same confidences, outcomes track them):
#   mean confidence = 0.82   accuracy = 0.80
#   reliability curve (per confidence bin):
#     bin           n   mean_conf   accuracy   gap
#     [0.6, 0.8)   3      0.633      0.667   0.033
#     [0.8, 1.0)   5      0.860      0.800   0.060
#     [1.0]        2      1.000      1.000   0.000
#   ECE   = weighted avg gap  = 0.040
#   Brier = mean (conf-out)^2 = 0.172
#
# ========================================================================
# What the two numbers caught:
#   ECE   0.420 -> 0.040   (gap between promise and reality shrinks 10.5x)
#   Brier 0.432 -> 0.172   (lower is better; rewards being right AND appropriately unsure)
# The over-confident model says 0.82 but is right 0.40 of the time:
# every bin sits below the diagonal. Confidence you cannot trust is
# worse than no confidence at all.
# ========================================================================
