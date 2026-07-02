"""
Evals from First Principles — Part 7: Is the Difference Real?

Model A scored 82%, Model B scored 84% on the SAME eval set. A real
improvement, or noise? Because both models were graded on the same items, the
comparison is PAIRED: we build the 2x2 of agreements/disagreements, then run
McNemar's test on only the two discordant cells (the items where the models
split). We derive the exact p-value by enumerating the binomial by hand, sanity
-check it with the chi-square approximation, and finish with a power/sample-size
back-of-envelope: how many items you would actually need to detect a 2-point
gain. Pure Python + NumPy, offline, deterministic (no SciPy; the normal tail is
computed from math.erfc and the exact p-value from math.comb).
"""

import math
import numpy as np

# A fixed seed makes the shuffled item order reproducible. The shuffle only
# changes which item sits at which index; it never changes the four cell counts.
RNG = np.random.default_rng(0)


# --- The task -------------------------------------------------------------
# Two models answer the SAME 50 questions. Each item is graded 1 (correct) or
# 0 (wrong) for each model. We construct the outcomes so that A gets 41/50 and
# B gets 42/50 (the 82% vs 84% headline), with the models splitting on 9 items.
def build_outcomes():
    """Return two aligned 0/1 arrays: a[i], b[i] grade the SAME item i."""
    # groups: (both right) x37, (A only) x4, (B only) x5, (both wrong) x4
    a = np.array([1] * 37 + [1] * 4 + [0] * 5 + [0] * 4)
    b = np.array([1] * 37 + [0] * 4 + [1] * 5 + [0] * 4)
    order = RNG.permutation(len(a))          # interleave, like a real run
    return a[order], b[order]


def paired_table(a, b):
    """The paired 2x2 counted by hand: (both_right, a_only, b_only, both_wrong)."""
    both_right = int(np.sum((a == 1) & (b == 1)))   # cell a
    a_only     = int(np.sum((a == 1) & (b == 0)))   # cell b: A right, B wrong
    b_only     = int(np.sum((a == 0) & (b == 1)))   # cell c: B right, A wrong
    both_wrong = int(np.sum((a == 0) & (b == 0)))   # cell d
    return both_right, a_only, b_only, both_wrong


def mcnemar_chi2(b, c):
    """Chi-square statistics on the discordant cells (uncorrected and Yates)."""
    uncorrected = (b - c) ** 2 / (b + c)
    yates = (abs(b - c) - 1) ** 2 / (b + c)         # continuity correction
    return uncorrected, yates


def chi2_1df_pvalue(stat):
    """Two-sided p for a chi-square(df=1): P(X>stat) = erfc(sqrt(stat/2))."""
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar_exact_p(b, c):
    """
    Exact two-sided McNemar p-value, enumerated by hand. Under H0 the count of
    discordant pairs favoring one model ~ Binomial(n=b+c, 0.5). The two-sided p
    is 2 * P(X <= min(b,c)), capped at 1.0.
    """
    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))   # C(n,0)+...+C(n,k)
    p = 2.0 * tail / (2 ** n)
    return min(1.0, p), n, k, tail


# --- Power / sample size --------------------------------------------------
# Standard normal critical values for alpha=0.05 two-sided and 80% power.
Z_ALPHA = 1.96      # P(|Z| > 1.96) = 0.05
Z_BETA = 0.84       # P(Z > 0.84)  = 0.20  -> 80% power


def sample_size_unpaired(pA, pB):
    """Two-proportion back-of-envelope: items PER MODEL to detect pB - pA."""
    var = pA * (1 - pA) + pB * (1 - pB)
    delta = pB - pA
    n = (Z_ALPHA + Z_BETA) ** 2 * var / delta ** 2
    return math.ceil(n)


def sample_size_paired(p_a_only, p_b_only):
    """
    Paired (McNemar) sample size, same coin-flip logic as the test itself.
    Discordant pairs are a coin with P(favor B) = rho; detecting rho vs 0.5 is a
    one-sample proportion test. Returns (discordant_pairs_needed, items_needed).
    """
    pi_d = p_a_only + p_b_only               # P(the two models disagree)
    rho = p_b_only / pi_d                     # among disagreements, share for B
    num = Z_ALPHA * math.sqrt(0.25) + Z_BETA * math.sqrt(rho * (1 - rho))
    m = num ** 2 / (rho - 0.5) ** 2           # discordant pairs needed
    return math.ceil(m), math.ceil(m / pi_d)


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 7 - IS THE DIFFERENCE REAL? paired comparison, McNemar, power.")
    print(line)

    a, b = build_outcomes()
    n = len(a)
    a_correct, b_correct = int(a.sum()), int(b.sum())
    print(f"Two models graded on the SAME {n}-item set (1 = correct, 0 = wrong).")
    print(f"  Model A: {a_correct}/{n} correct = {100*a_correct/n:.1f}%")
    print(f"  Model B: {b_correct}/{n} correct = {100*b_correct/n:.1f}%")
    print(f"  Headline gap: B - A = +{100*(b_correct-a_correct)/n:.1f} points. Real, or noise?")

    # The signal lives in the items where the two models SPLIT. List them.
    disagree = [i for i in range(n) if a[i] != b[i]]
    print(f"\nThe {len(disagree)} items where the models split (the rest, both agree):")
    print("  item   A   B   who won")
    for i in disagree:
        who = "A only" if a[i] == 1 else "B only"
        print(f"   {i:>3}   {a[i]}   {b[i]}   {who}")

    both_right, a_only, b_only, both_wrong = paired_table(a, b)
    print("\n" + "-" * 72)
    print("Step 1 - the paired 2x2 (every item lands in exactly one cell):")
    print("-" * 72)
    print("                     B correct   B wrong")
    print(f"  A correct              {both_right:>2}         {a_only:>2}      (a, b)")
    print(f"  A wrong                 {b_only:>2}          {both_wrong:>2}      (c, d)")
    print(f"  cells: a={both_right}  b={a_only}  c={b_only}  d={both_wrong}   (n={n})")
    print(f"  Only the DISCORDANT cells carry the signal: b={a_only} (A right, B wrong)")
    print(f"  and c={b_only} (B right, A wrong). The {both_right} both-right and {both_wrong} both-wrong")
    print("  items say nothing about which model is better.")

    print("\n" + "-" * 72)
    print(f"Step 2 - McNemar's test on the discordant cells (b={a_only}, c={b_only}):")
    print("-" * 72)
    uncorr, yates = mcnemar_chi2(a_only, b_only)
    p_chi = chi2_1df_pvalue(uncorr)
    p_exact, disc_n, k, tail = mcnemar_exact_p(a_only, b_only)
    print("  Under H0 (models equal), each of the b+c disagreements is a coin")
    print(f"  flip: the count going B's way ~ Binomial({disc_n}, 0.5).")
    print(f"  chi-square (uncorrected) = (b-c)^2/(b+c) = ({a_only}-{b_only})^2/{a_only+b_only} = {uncorr:.3f}")
    print(f"  chi-square (Yates corr.) = (|b-c|-1)^2/(b+c) = 0^2/{a_only+b_only} = {yates:.3f}")
    print(f"  approx p (chi-square, 1 df) = erfc(sqrt({uncorr:.3f}/2)) = {p_chi:.3f}")
    print(f"  exact p: 2 * [C({disc_n},0)+...+C({disc_n},{k})] / 2^{disc_n}")
    print(f"         = 2 * {tail} / {2**disc_n} = {p_exact:.3f}")
    print(f"  -> p = {p_exact:.3f} >> 0.05. A {b_only}-vs-{a_only} split over {disc_n} disagreements is")
    print("     exactly what a fair coin looks like. NOT significant: on this")
    print("     set we cannot rule out noise. The +2 points is not yet a result.")

    print("\n" + "-" * 72)
    print("Step 3 - power: how big a set to detect a TRUE 2-point gain?")
    print("-" * 72)
    pA, pB = a_correct / n, b_correct / n
    n_unpaired = sample_size_unpaired(pA, pB)
    disc_needed, n_paired = sample_size_paired(a_only / n, b_only / n)
    print(f"  Target: detect pB-pA = {pB-pA:.2f} at alpha=0.05 (z={Z_ALPHA}), 80% power (z={Z_BETA}).")
    print(f"  Unpaired two-proportion rule:")
    print(f"    n = (z_a+z_b)^2 * [pA(1-pA)+pB(1-pB)] / (pB-pA)^2")
    print(f"      = {(Z_ALPHA+Z_BETA)**2:.2f} * {pA*(1-pA)+pB*(1-pB):.4f} / {(pB-pA)**2:.4f} = {n_unpaired} items PER MODEL.")
    print(f"  Paired (McNemar) rule, same coin logic, {100*(a_only+b_only)/n:.0f}% of items discordant:")
    print(f"    need {disc_needed} discordant pairs -> {n_paired} items total.")
    print(f"  -> Thousands of items to catch a 2-point gap 80% of the time. Our")
    print(f"     {n}-item set had almost no chance. Pairing helps ({n_paired} < {n_unpaired}),")
    print("     but a 2-point headline on a small eval is a vibe, not a result.")
    print(line)

# Expected output:
# ========================================================================
# PART 7 - IS THE DIFFERENCE REAL? paired comparison, McNemar, power.
# ========================================================================
# Two models graded on the SAME 50-item set (1 = correct, 0 = wrong).
#   Model A: 41/50 correct = 82.0%
#   Model B: 42/50 correct = 84.0%
#   Headline gap: B - A = +2.0 points. Real, or noise?
#
# The 9 items where the models split (the rest, both agree):
#   item   A   B   who won
#     15   0   1   B only
#     20   1   0   A only
#     22   0   1   B only
#     24   1   0   A only
#     34   0   1   B only
#     36   1   0   A only
#     37   0   1   B only
#     41   1   0   A only
#     46   0   1   B only
#
# ------------------------------------------------------------------------
# Step 1 - the paired 2x2 (every item lands in exactly one cell):
# ------------------------------------------------------------------------
#                      B correct   B wrong
#   A correct              37          4      (a, b)
#   A wrong                  5           4      (c, d)
#   cells: a=37  b=4  c=5  d=4   (n=50)
#   Only the DISCORDANT cells carry the signal: b=4 (A right, B wrong)
#   and c=5 (B right, A wrong). The 37 both-right and 4 both-wrong
#   items say nothing about which model is better.
#
# ------------------------------------------------------------------------
# Step 2 - McNemar's test on the discordant cells (b=4, c=5):
# ------------------------------------------------------------------------
#   Under H0 (models equal), each of the b+c disagreements is a coin
#   flip: the count going B's way ~ Binomial(9, 0.5).
#   chi-square (uncorrected) = (b-c)^2/(b+c) = (4-5)^2/9 = 0.111
#   chi-square (Yates corr.) = (|b-c|-1)^2/(b+c) = 0^2/9 = 0.000
#   approx p (chi-square, 1 df) = erfc(sqrt(0.111/2)) = 0.739
#   exact p: 2 * [C(9,0)+...+C(9,4)] / 2^9
#          = 2 * 256 / 512 = 1.000
#   -> p = 1.000 >> 0.05. A 5-vs-4 split over 9 disagreements is
#      exactly what a fair coin looks like. NOT significant: on this
#      set we cannot rule out noise. The +2 points is not yet a result.
#
# ------------------------------------------------------------------------
# Step 3 - power: how big a set to detect a TRUE 2-point gain?
# ------------------------------------------------------------------------
#   Target: detect pB-pA = 0.02 at alpha=0.05 (z=1.96), 80% power (z=0.84).
#   Unpaired two-proportion rule:
#     n = (z_a+z_b)^2 * [pA(1-pA)+pB(1-pB)] / (pB-pA)^2
#       = 7.84 * 0.2820 / 0.0004 = 5528 items PER MODEL.
#   Paired (McNemar) rule, same coin logic, 18% of items discordant:
#     need 633 discordant pairs -> 3515 items total.
#   -> Thousands of items to catch a 2-point gap 80% of the time. Our
#      50-item set had almost no chance. Pairing helps (3515 < 5528),
#      but a 2-point headline on a small eval is a vibe, not a result.
# ========================================================================
