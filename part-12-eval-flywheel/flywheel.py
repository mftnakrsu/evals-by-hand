"""
Evals from First Principles - Part 12: The Eval Flywheel

Your best eval set is your production logs. We take 12 mock production traces
from a support assistant. Each trace carries CHEAP signals we get for free (the
model's self-reported confidence and a guardrail / thumbs-down flag) but its
true label is UNKNOWN until a human pays to annotate it. We rank the traces by a
priority score, spend a small annotation budget on the most suspicious ones,
reveal their labels, and watch the golden set grow exactly where the model is
weakest. Then we count why hard-case sampling fills the golden set with failures
about 2x faster than random sampling. Pure Python, offline, deterministic.
"""

import random
from collections import namedtuple

Trace = namedtuple("Trace", "id question confidence flag correct")

# --- The production log ----------------------------------------------------
# 12 traces served by a support assistant. confidence = the model's self-report
# (cheap, always visible). flag = a guardrail hit or a thumbs-down (cheap, always
# visible). correct = the HIDDEN ground truth (1 = good answer, 0 = a failure)
# that a human reveals ONLY when we spend budget to annotate that trace.
TRACES = [
    Trace("T01", "How do I reset my password?",        0.96, 0, 1),
    Trace("T02", "What are your business hours?",       0.93, 0, 1),
    Trace("T03", "Where is my refund?",                 0.91, 0, 1),
    Trace("T04", "Change my shipping address",          0.88, 0, 1),
    Trace("T05", "Cancel my subscription",              0.84, 0, 1),
    Trace("T06", "Do you ship to Canada?",              0.80, 0, 1),
    Trace("T07", "Is the pro plan tax-deductible?",     0.74, 0, 1),
    Trace("T08", "Why was I charged twice?",            0.68, 0, 0),
    Trace("T09", "Explain your data-retention policy",  0.62, 1, 0),
    Trace("T10", "Can I get a discount?",               0.55, 1, 1),
    Trace("T11", "Is my warranty still valid?",         0.48, 1, 0),
    Trace("T12", "Reactivate my closed account",        0.40, 1, 0),
]


def priority(t):
    """A cheap suspicion score: low confidence and a flag both push it up.

    priority = (1 - confidence) + 0.5 * flag. Needs NO annotation to compute."""
    return (1 - t.confidence) + 0.5 * t.flag


def n_fail(traces):
    """Count the true failures (correct == 0) in a batch of annotated traces."""
    return sum(1 for t in traces if t.correct == 0)


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 12 - THE EVAL FLYWHEEL: mine production logs into a golden set.")
    print(line)

    total = len(TRACES)
    true_fail = n_fail(TRACES)  # ground truth, unknown to us until we annotate

    # --- 1. The log, with the cheap signals we get for free ----------------
    print("The production log: 12 traces. The cheap signals (conf, flag) are")
    print("free; the true label costs a human annotation, so it starts hidden.")
    print()
    print("  id    conf  flag  priority  true label")
    for t in TRACES:
        print(f"  {t.id}  {t.confidence:>5.2f}  {t.flag:>3}  {priority(t):>7.2f}   (hidden)")

    # --- 2. Rank by the cheap priority signal ------------------------------
    ranked = sorted(TRACES, key=priority, reverse=True)
    print()
    print("-" * 72)
    print("Rank by priority (highest = most suspicious). No labels needed yet:")
    print("  " + "  ".join(t.id for t in ranked))

    # --- 3. Spend one annotation budget: hard-case vs random ---------------
    budget = 6
    print()
    print("-" * 72)
    print(f"Spend a budget of {budget} annotations. Which traces do we send?")
    print()

    # Hard-case sampling: annotate the top-`budget` by priority.
    hard = ranked[:budget]
    hard_fail = n_fail(hard)  # labels revealed by annotation
    print(f"  Hard-case batch (top {budget} by priority):")
    print(f"    {' '.join(t.id for t in hard)}")
    print(f"    failures found = {hard_fail}/{budget}   "
          f"(failure density = {hard_fail / budget:.2f})")
    print(f"    failure coverage = {hard_fail}/{true_fail} of all real failures "
          f"= {hard_fail / true_fail:.2f}")

    # Random sampling: a fixed-seed draw stands in for "just grab some logs".
    rnd = random.Random(0).sample(TRACES, budget)
    rnd = sorted(rnd, key=lambda t: t.id)
    rnd_fail = n_fail(rnd)
    exp_fail = budget * true_fail / total  # exact expectation, no simulation
    print()
    print(f"  Random batch (seed 0):")
    print(f"    {' '.join(t.id for t in rnd)}")
    print(f"    failures found = {rnd_fail}/{budget}   "
          f"(failure density = {rnd_fail / budget:.2f})")
    print(f"    expected failures for a random batch = {budget}*{true_fail}/{total} "
          f"= {exp_fail:.2f}")
    print(f"    failure coverage = {rnd_fail}/{true_fail} of all real failures "
          f"= {rnd_fail / true_fail:.2f}")

    print()
    print(f"  -> Same {budget} annotations: hard-case sampling caught ALL "
          f"{true_fail} failures,")
    print(f"     random caught {rnd_fail}. The golden set is {hard_fail / budget:.2f} "
          f"failures vs the {true_fail / total:.2f} base rate: ~2x denser.")

    # --- 4. The flywheel: annotate in rounds, watch the golden set grow -----
    print()
    print("-" * 72)
    print("The flywheel: annotate 2 hard cases per round; the golden set grows")
    print("where the model is weakest, round by round.")
    print()
    print("  round  new       new fail  golden  cum fail  coverage  gold rate")
    golden, cum = [], 0
    per_round = 2
    for r in range(3):
        batch = ranked[r * per_round:(r + 1) * per_round]
        nf = n_fail(batch)
        golden += batch
        cum += nf
        ids = "+".join(t.id for t in batch)
        print(f"   {r + 1}     {ids}  {nf:>7}  {len(golden):>6}  {cum:>7}   "
              f"{cum / true_fail:>6.2f}    {cum / len(golden):>6.2f}")

    # --- 5. Close the loop honestly: enriched != production rate -----------
    print()
    print("-" * 72)
    print("Closing the loop (honestly):")
    print(f"  The golden set is deliberately failure-dense "
          f"({cum}/{len(golden)} = {cum / len(golden):.2f}); that is the point,")
    print("  it guards the model's weak spots. But that rate is NOT production's:")
    print(f"  the true production failure rate is {true_fail}/{total} "
          f"= {true_fail / total:.2f}. To estimate")
    print("  THAT unbiased, keep a small random holdout alongside the mined set.")
    print(line)

# Expected output:
# ========================================================================
# PART 12 - THE EVAL FLYWHEEL: mine production logs into a golden set.
# ========================================================================
# The production log: 12 traces. The cheap signals (conf, flag) are
# free; the true label costs a human annotation, so it starts hidden.
#
#   id    conf  flag  priority  true label
#   T01   0.96    0     0.04   (hidden)
#   T02   0.93    0     0.07   (hidden)
#   T03   0.91    0     0.09   (hidden)
#   T04   0.88    0     0.12   (hidden)
#   T05   0.84    0     0.16   (hidden)
#   T06   0.80    0     0.20   (hidden)
#   T07   0.74    0     0.26   (hidden)
#   T08   0.68    0     0.32   (hidden)
#   T09   0.62    1     0.88   (hidden)
#   T10   0.55    1     0.95   (hidden)
#   T11   0.48    1     1.02   (hidden)
#   T12   0.40    1     1.10   (hidden)
#
# ------------------------------------------------------------------------
# Rank by priority (highest = most suspicious). No labels needed yet:
#   T12  T11  T10  T09  T08  T07  T06  T05  T04  T03  T02  T01
#
# ------------------------------------------------------------------------
# Spend a budget of 6 annotations. Which traces do we send?
#
#   Hard-case batch (top 6 by priority):
#     T12 T11 T10 T09 T08 T07
#     failures found = 4/6   (failure density = 0.67)
#     failure coverage = 4/4 of all real failures = 1.00
#
#   Random batch (seed 0):
#     T01 T04 T05 T07 T08 T12
#     failures found = 2/6   (failure density = 0.33)
#     expected failures for a random batch = 6*4/12 = 2.00
#     failure coverage = 2/4 of all real failures = 0.50
#
#   -> Same 6 annotations: hard-case sampling caught ALL 4 failures,
#      random caught 2. The golden set is 0.67 failures vs the 0.33 base rate: ~2x denser.
#
# ------------------------------------------------------------------------
# The flywheel: annotate 2 hard cases per round; the golden set grows
# where the model is weakest, round by round.
#
#   round  new       new fail  golden  cum fail  coverage  gold rate
#    1     T12+T11        2       2        2     0.50      1.00
#    2     T10+T09        1       4        3     0.75      0.75
#    3     T08+T07        1       6        4     1.00      0.67
#
# ------------------------------------------------------------------------
# Closing the loop (honestly):
#   The golden set is deliberately failure-dense (4/6 = 0.67); that is the point,
#   it guards the model's weak spots. But that rate is NOT production's:
#   the true production failure rate is 4/12 = 0.33. To estimate
#   THAT unbiased, keep a small random holdout alongside the mined set.
# ========================================================================
