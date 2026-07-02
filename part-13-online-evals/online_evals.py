"""
Evals from First Principles - Part 13: Online Evals.

Offline scores are necessary, not sufficient. The only place a change proves
itself is production, under a controlled A/B test. Here we run one by hand:
two arms (control vs treatment) of a support chatbot, a primary business
metric (resolution rate) with an observed lift, a two-proportion z-test and
p-value derived by hand, and a GUARDRAIL metric (refusal rate) that must not
regress even when the primary metric wins. The final ship/no-ship verdict
needs BOTH: a significant primary gain AND an intact guardrail.

Pure Python + stdlib math, offline, deterministic. The normal CDF is computed
from math.erf (exact, no SciPy). A silent SciPy cross-check runs only if SciPy
happens to be installed; it never changes what is printed.
"""

import math

# --- The experiment -------------------------------------------------------
# We shipped a new answer-generation prompt to a support chatbot and split
# live traffic 50/50 for a week. Every session is one trial.
#   PRIMARY metric  = resolution rate: did the bot resolve the issue without
#                     escalating to a human? (higher is better)
#   GUARDRAIL metric = refusal rate: did the bot wrongly say "I can't help"?
#                      (lower is better; a business guardrail, not an offline
#                       score - you only see it in production)
#
# Counts are the tiny, traceable heart of the whole test:
#   arm         sessions  resolved  refused
CONTROL   = dict(name="control  (old prompt)", n=400, resolved=168, refused=20)
TREATMENT = dict(name="treatment(new prompt)", n=400, resolved=200, refused=40)

ALPHA = 0.05            # significance level for the primary metric
GUARDRAIL_MARGIN = 0.02  # treatment refusal rate may rise at most +2 pp


def rate(count, n):
    """A conversion-style proportion: successes / trials."""
    return count / n if n else 0.0


def normal_cdf(z):
    """Standard-normal CDF via the error function. Phi(z)=0.5(1+erf(z/sqrt2))."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_ztest(x1, n1, x2, n2):
    """
    Two-proportion z-test, every step by hand.
      p1, p2   : the two observed rates
      p_pool   : rate if the arms were identical (all successes / all trials)
      se       : standard error of (p2 - p1) under that pooled null
      z        : (p2 - p1) / se
      p_value  : two-sided, 2 * P(Z > |z|)
    Returns a dict of the intermediate numbers so the caller can print them.
    """
    p1 = rate(x1, n1)
    p2 = rate(x2, n2)
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z = (p2 - p1) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return dict(p1=p1, p2=p2, p_pool=p_pool, se=se, z=z, p_value=p_value)


def bar(char="=", width=72):
    return char * width


if __name__ == "__main__":
    print(bar())
    print("PART 13 - ONLINE EVALS: an A/B test with a guardrail, by hand.")
    print(bar())

    # --- The two arms -----------------------------------------------------
    print("Two arms of live traffic, one week, 50/50 split:")
    for arm in (CONTROL, TREATMENT):
        print(f"  {arm['name']}:  n={arm['n']}  "
              f"resolved={arm['resolved']}  refused={arm['refused']}")

    # --- Primary metric: resolution rate ----------------------------------
    p_c = rate(CONTROL["resolved"], CONTROL["n"])
    p_t = rate(TREATMENT["resolved"], TREATMENT["n"])
    abs_lift = p_t - p_c
    rel_lift = abs_lift / p_c

    print("\n" + bar("-"))
    print("PRIMARY METRIC - resolution rate (higher is better)")
    print(bar("-"))
    print(f"  control   = {CONTROL['resolved']}/{CONTROL['n']} = {p_c:.3f}")
    print(f"  treatment = {TREATMENT['resolved']}/{TREATMENT['n']} = {p_t:.3f}")
    print(f"  absolute lift = {p_t:.3f} - {p_c:.3f} = {abs_lift:+.3f}  "
          f"({abs_lift*100:+.1f} pp)")
    print(f"  relative lift = {abs_lift:.3f}/{p_c:.3f} = {rel_lift*100:+.1f}%")

    # --- Is the lift real? two-proportion z-test --------------------------
    t = two_proportion_ztest(CONTROL["resolved"], CONTROL["n"],
                             TREATMENT["resolved"], TREATMENT["n"])
    print("\n  Is it real, or noise? two-proportion z-test:")
    print(f"    pooled rate p = ({CONTROL['resolved']}+{TREATMENT['resolved']})"
          f"/({CONTROL['n']}+{TREATMENT['n']}) = {t['p_pool']:.3f}")
    print(f"    SE = sqrt(p(1-p)(1/n_c + 1/n_t)) = {t['se']:.5f}")
    print(f"    z  = (p_t - p_c)/SE = {abs_lift:+.3f}/{t['se']:.5f} = {t['z']:.3f}")
    print(f"    p-value (two-sided) = {t['p_value']:.4f}")
    primary_sig = t["p_value"] < ALPHA
    verdict = "SIGNIFICANT" if primary_sig else "not significant"
    print(f"    at alpha={ALPHA}: {verdict}  (p {'<' if primary_sig else '>='} {ALPHA})")

    # --- Guardrail metric: refusal rate -----------------------------------
    r_c = rate(CONTROL["refused"], CONTROL["n"])
    r_t = rate(TREATMENT["refused"], TREATMENT["n"])
    guard_delta = r_t - r_c
    guard_ok = guard_delta <= GUARDRAIL_MARGIN

    print("\n" + bar("-"))
    print("GUARDRAIL METRIC - refusal rate (must NOT regress, lower is better)")
    print(bar("-"))
    print(f"  control   = {CONTROL['refused']}/{CONTROL['n']} = {r_c:.3f}")
    print(f"  treatment = {TREATMENT['refused']}/{TREATMENT['n']} = {r_t:.3f}")
    print(f"  change = {r_t:.3f} - {r_c:.3f} = {guard_delta:+.3f}  "
          f"({guard_delta*100:+.1f} pp)")
    print(f"  allowed margin = +{GUARDRAIL_MARGIN*100:.0f} pp  ->  "
          f"{'OK' if guard_ok else 'BREACHED'}")

    # --- The ship / no-ship verdict ---------------------------------------
    ship = primary_sig and guard_ok
    print("\n" + bar())
    print("VERDICT: ship only if primary is SIGNIFICANT and guardrail is OK.")
    print(f"  primary significant? {primary_sig}   guardrail ok? {guard_ok}")
    if ship:
        print("  -> SHIP IT.")
    else:
        print("  -> DO NOT SHIP. The new prompt resolves more issues (a real,")
        print("     significant +8 pp gain) but refuses twice as often, breaching")
        print("     the guardrail. An offline win is necessary, not sufficient.")
    print(bar())

    # Silent verification: if SciPy is present, its z-test must agree. This
    # never prints, so the expected-output block is identical with or without it.
    try:
        from scipy.stats import norm
        assert abs(t["p_value"] - 2 * (1 - norm.cdf(abs(t["z"])))) < 1e-12
    except ImportError:
        pass

# Expected output:
# ========================================================================
# PART 13 - ONLINE EVALS: an A/B test with a guardrail, by hand.
# ========================================================================
# Two arms of live traffic, one week, 50/50 split:
#   control  (old prompt):  n=400  resolved=168  refused=20
#   treatment(new prompt):  n=400  resolved=200  refused=40
#
# ------------------------------------------------------------------------
# PRIMARY METRIC - resolution rate (higher is better)
# ------------------------------------------------------------------------
#   control   = 168/400 = 0.420
#   treatment = 200/400 = 0.500
#   absolute lift = 0.500 - 0.420 = +0.080  (+8.0 pp)
#   relative lift = 0.080/0.420 = +19.0%
#
#   Is it real, or noise? two-proportion z-test:
#     pooled rate p = (168+200)/(400+400) = 0.460
#     SE = sqrt(p(1-p)(1/n_c + 1/n_t)) = 0.03524
#     z  = (p_t - p_c)/SE = +0.080/0.03524 = 2.270
#     p-value (two-sided) = 0.0232
#     at alpha=0.05: SIGNIFICANT  (p < 0.05)
#
# ------------------------------------------------------------------------
# GUARDRAIL METRIC - refusal rate (must NOT regress, lower is better)
# ------------------------------------------------------------------------
#   control   = 20/400 = 0.050
#   treatment = 40/400 = 0.100
#   change = 0.100 - 0.050 = +0.050  (+5.0 pp)
#   allowed margin = +2 pp  ->  BREACHED
#
# ========================================================================
# VERDICT: ship only if primary is SIGNIFICANT and guardrail is OK.
#   primary significant? True   guardrail ok? False
#   -> DO NOT SHIP. The new prompt resolves more issues (a real,
#      significant +8 pp gain) but refuses twice as often, breaching
#      the guardrail. An offline win is necessary, not sufficient.
# ========================================================================
