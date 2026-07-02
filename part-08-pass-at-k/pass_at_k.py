"""
Evals from First Principles — Part 8: pass@k.

The code-gen / agent metric everyone quotes and few compute correctly.
You sample n candidate solutions, c of them pass the unit tests, and you
want to report pass@k: the chance that at least one of k independent draws
is correct. The obvious plug-in 1-(1-c/n)^k is BIASED. The estimator used
by the Codex paper is an exact combinatorial identity,

    pass@k = 1 - C(n-c, k) / C(n, k),

which we derive term by term and then CHECK against a brute-force count over
every size-k subset at n=5, c=2, k=2. Pure Python, offline, deterministic:
no sampling at all, so every number is exact and reproducible.
"""

from math import comb
from itertools import combinations
from fractions import Fraction

# --- The scenario ---------------------------------------------------------
# One coding task. We ask the model for n=5 completions and run each against
# a hidden unit-test suite. c=2 of the 5 pass. We label the samples so the
# reader can see exactly which ones are correct.
N = 5                        # samples generated for the task
C = 2                        # of those, how many pass the tests
K = 2                        # budget: we would ship the best of k draws
CORRECT = [1, 1, 0, 0, 0]    # sample s0,s1 pass; s2,s3,s4 fail. sum == C


def plugin_pass_at_k(n, c, k):
    """The tempting WRONG estimator: pretend each draw is an independent coin
    with success rate p_hat = c/n, so 'miss all k' = (1 - c/n)^k. This models
    sampling WITH replacement and is biased for the finite pool we actually
    drew."""
    return 1.0 - (1.0 - c / n) ** k


def unbiased_pass_at_k(n, c, k):
    """The Codex-paper estimator. Of C(n,k) equally likely size-k subsets of
    the n samples, exactly C(n-c, k) contain none of the c correct ones (you
    must draw all k from the n-c wrong ones). So the chance of an all-wrong
    draw is C(n-c,k)/C(n,k), and pass@k is one minus that. When k > n-c,
    C(n-c,k)=0 and the estimator is exactly 1.0, as it must be."""
    return 1.0 - comb(n - c, k) / comb(n, k)


def brute_force_pass_at_k(correct_flags, k):
    """Ground truth by enumeration: list EVERY size-k subset of the samples,
    count how many contain at least one correct sample. Returns (passed,
    total). This is what the estimator is an estimator OF."""
    idx = range(len(correct_flags))
    passed = total = 0
    for subset in combinations(idx, k):
        total += 1
        if any(correct_flags[i] for i in subset):
            passed += 1
    return passed, total


# --- Exact bias demonstration (no Monte Carlo) ----------------------------
# We can PROVE the estimators biased/unbiased without sampling by taking the
# exact expectation over the distribution of c. If each sample passes
# independently with true rate p, then c ~ Binomial(n, p), and the true
# quantity we want to estimate is pass@k = 1 - (1-p)^k. We average each
# estimator over all c = 0..n weighted by its binomial probability, in exact
# rational arithmetic, and compare to the truth.
def binom_pmf(c, n, p):
    """P(exactly c successes in n) as an exact Fraction."""
    return comb(n, c) * p ** c * (1 - p) ** (n - c)


def unbiased_frac(n, c, k):
    return 1 - Fraction(comb(n - c, k), comb(n, k))


def plugin_frac(n, c, k):
    return 1 - (1 - Fraction(c, n)) ** k


def expected_over_c(estimator_frac, n, k, p):
    """Exact E[estimator] with c ~ Binomial(n, p), as a Fraction."""
    return sum((binom_pmf(c, n, p) * estimator_frac(n, c, k)
                for c in range(n + 1)), Fraction(0))


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 8 - pass@k: the unbiased estimator as a combinatorial identity.")
    print(line)

    labels = " ".join(f"s{i}({'ok' if CORRECT[i] else 'x '})" for i in range(N))
    print(f"One task, n={N} sampled completions, c={C} pass the unit tests, k={K}.")
    print(f"  samples:  {labels}")
    print(f"  p_hat = c/n = {C}/{N} = {C / N:.2f}   (empirical per-sample pass rate)")

    print("\n" + "-" * 72)
    print("STEP 1 - the tempting plug-in (WITH replacement, and biased):")
    print("-" * 72)
    p_hat = C / N
    plug = plugin_pass_at_k(N, C, K)
    print(f"  pass@{K} ~= 1 - (1 - c/n)^k = 1 - (1 - {p_hat:.2f})^{K}"
          f" = 1 - {(1 - p_hat) ** K:.2f} = {plug:.3f}")
    print("  It treats the 5-sample pool as an infinite coin. We only drew 5.")

    print("\n" + "-" * 72)
    print("STEP 2 - the unbiased estimator (WITHOUT replacement):")
    print("-" * 72)
    all_wrong = comb(N - C, K)
    all_subsets = comb(N, K)
    unb = unbiased_pass_at_k(N, C, K)
    print(f"  total size-k subsets      C(n, k)   = C({N},{K}) = {all_subsets}")
    print(f"  all-wrong size-k subsets  C(n-c, k) = C({N - C},{K}) = {all_wrong}"
          f"   (both picks from the {N - C} failing samples)")
    print(f"  P(draw is all wrong) = C(n-c,k)/C(n,k) = {all_wrong}/{all_subsets}"
          f" = {all_wrong / all_subsets:.3f}")
    print(f"  pass@{K} = 1 - C(n-c,k)/C(n,k) = 1 - {all_wrong}/{all_subsets}"
          f" = {unb:.3f}")

    print("\n" + "-" * 72)
    print("STEP 3 - brute-force CHECK: enumerate every size-k subset at n=5.")
    print("-" * 72)
    passed, total = brute_force_pass_at_k(CORRECT, K)
    print(f"  size-{K} subsets of {{s0..s{N - 1}}} that contain >=1 correct sample:")
    for subset in combinations(range(N), K):
        names = "{" + ",".join(f"s{i}" for i in subset) + "}"
        hit = any(CORRECT[i] for i in subset)
        print(f"    {names:11s} -> {'PASS' if hit else 'fail'}")
    print(f"  passing subsets / total = {passed}/{total} = {passed / total:.3f}")
    print(f"  estimator said {unb:.3f}.  Match: {abs(passed / total - unb) < 1e-12}")

    print("\n" + "-" * 72)
    print("STEP 4 - pass@k for every budget k (same n=5, c=2):")
    print("-" * 72)
    print("    k   plug-in(biased)   unbiased    brute-force")
    for k in range(1, N + 1):
        pl = plugin_pass_at_k(N, C, k)
        ub = unbiased_pass_at_k(N, C, k)
        pk, tk = brute_force_pass_at_k(CORRECT, k)
        print(f"    {k}      {pl:.3f}          {ub:.3f}       {pk}/{tk} = {pk / tk:.3f}")
    print("  pass@k rises with k and hits 1.000 once k > n-c (can't miss all correct).")

    print("\n" + "-" * 72)
    print("STEP 5 - WHY 'unbiased': exact expectation over c ~ Binomial(n, p).")
    print("-" * 72)
    p = Fraction(2, 5)                     # true per-sample pass rate = 0.40
    true = 1 - (1 - p) ** K                # true pass@k = 1 - (1-p)^k
    e_unb = expected_over_c(unbiased_frac, N, K, p)
    e_plug = expected_over_c(plugin_frac, N, K, p)
    print(f"  true per-sample rate p = {float(p):.2f},  true pass@{K}"
          f" = 1 - (1-p)^{K} = {float(true):.3f}")
    print(f"  E[unbiased estimator] = {float(e_unb):.3f}"
          f"   (exact: {e_unb})  -> matches truth")
    print(f"  E[plug-in  estimator] = {float(e_plug):.3f}"
          f"   (exact: {e_plug})  -> low by {float(true - e_plug):.3f}")
    print("  The plug-in systematically UNDER-reports pass@k; the combinatorial")
    print("  estimator is exactly right on average. Report the unbiased one.")
    print(line)

# Expected output:
# ========================================================================
# PART 8 - pass@k: the unbiased estimator as a combinatorial identity.
# ========================================================================
# One task, n=5 sampled completions, c=2 pass the unit tests, k=2.
#   samples:  s0(ok) s1(ok) s2(x ) s3(x ) s4(x )
#   p_hat = c/n = 2/5 = 0.40   (empirical per-sample pass rate)
#
# ------------------------------------------------------------------------
# STEP 1 - the tempting plug-in (WITH replacement, and biased):
# ------------------------------------------------------------------------
#   pass@2 ~= 1 - (1 - c/n)^k = 1 - (1 - 0.40)^2 = 1 - 0.36 = 0.640
#   It treats the 5-sample pool as an infinite coin. We only drew 5.
#
# ------------------------------------------------------------------------
# STEP 2 - the unbiased estimator (WITHOUT replacement):
# ------------------------------------------------------------------------
#   total size-k subsets      C(n, k)   = C(5,2) = 10
#   all-wrong size-k subsets  C(n-c, k) = C(3,2) = 3   (both picks from the 3 failing samples)
#   P(draw is all wrong) = C(n-c,k)/C(n,k) = 3/10 = 0.300
#   pass@2 = 1 - C(n-c,k)/C(n,k) = 1 - 3/10 = 0.700
#
# ------------------------------------------------------------------------
# STEP 3 - brute-force CHECK: enumerate every size-k subset at n=5.
# ------------------------------------------------------------------------
#   size-2 subsets of {s0..s4} that contain >=1 correct sample:
#     {s0,s1}     -> PASS
#     {s0,s2}     -> PASS
#     {s0,s3}     -> PASS
#     {s0,s4}     -> PASS
#     {s1,s2}     -> PASS
#     {s1,s3}     -> PASS
#     {s1,s4}     -> PASS
#     {s2,s3}     -> fail
#     {s2,s4}     -> fail
#     {s3,s4}     -> fail
#   passing subsets / total = 7/10 = 0.700
#   estimator said 0.700.  Match: True
#
# ------------------------------------------------------------------------
# STEP 4 - pass@k for every budget k (same n=5, c=2):
# ------------------------------------------------------------------------
#     k   plug-in(biased)   unbiased    brute-force
#     1      0.400          0.400       2/5 = 0.400
#     2      0.640          0.700       7/10 = 0.700
#     3      0.784          0.900       9/10 = 0.900
#     4      0.870          1.000       5/5 = 1.000
#     5      0.922          1.000       1/1 = 1.000
#   pass@k rises with k and hits 1.000 once k > n-c (can't miss all correct).
#
# ------------------------------------------------------------------------
# STEP 5 - WHY 'unbiased': exact expectation over c ~ Binomial(n, p).
# ------------------------------------------------------------------------
#   true per-sample rate p = 0.40,  true pass@2 = 1 - (1-p)^2 = 0.640
#   E[unbiased estimator] = 0.640   (exact: 16/25)  -> matches truth
#   E[plug-in  estimator] = 0.592   (exact: 74/125)  -> low by 0.048
#   The plug-in systematically UNDER-reports pass@k; the combinatorial
#   estimator is exactly right on average. Report the unbiased one.
# ========================================================================
