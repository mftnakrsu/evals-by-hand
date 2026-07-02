"""
Evals from First Principles - Part 5: Judging the Judge.

An LLM judge is itself a model, so it has systematic, MEASURABLE biases. We
build a deterministic mock judge with two baked-in flaws: a POSITION bias (it
gives the answer shown first, in slot A, a fixed bonus) and a VERBOSITY bias
(it rewards longer answers regardless of quality). Then we measure each on a
dozen paired A-vs-B comparisons, and CORRECT the position bias by running both
orderings and averaging, watching the disagreement-with-truth rate fall.

The judge scores each answer as:

    score = quality + (POS_BIAS if shown in slot A else 0) + VERB_PER_WORD * words

A fair judge would use `quality` alone. Ours does not. Everything below is pure
Python (no imports), offline, and deterministic: no randomness, no API key.

(Self-preference - a bonus the judge hands to answers from its own model - is
the exact same recipe with a different trigger; measure and correct it the same
way. We keep the runnable demo to position + verbosity to stay concrete.)
"""

from collections import namedtuple

# --- The judge's two baked-in biases -------------------------------------
# POS_BIAS: a full quality-point handed to whichever answer sits in slot A.
# VERB_PER_WORD: extra credit per word, so a wordier answer creeps ahead.
POS_BIAS = 1.0
VERB_PER_WORD = 0.02

# --- The paired comparisons ----------------------------------------------
# Each pair pits system X against system Y on one prompt. An answer is just its
# (quality, words): quality is the honest score a fair judge would use; words is
# its length. The TRUE winner is simply the higher-quality answer.
Pair = namedtuple("Pair", "id xq xw yq yw")

PAIRS = [
    Pair("p01", 5, 40, 6, 20),   # Y better, X wordier  -> position tips it to X
    Pair("p02", 6, 50, 7, 25),   # Y better, X wordier
    Pair("p03", 3, 45, 4, 30),   # Y better, X wordier
    Pair("p04", 7, 35, 8, 25),   # Y better, X wordier
    Pair("p05", 9, 30, 4, 40),   # X clearly better
    Pair("p06", 8, 20, 5, 30),   # X clearly better
    Pair("p07", 3, 30, 8, 20),   # Y clearly better
    Pair("p08", 4, 25, 7, 35),   # Y clearly better
    Pair("p09", 7, 25, 6, 40),   # X better, Y wordier  -> order-sensitive
    Pair("p10", 8, 20, 7, 45),   # X better, Y wordier  -> order-sensitive
    Pair("p11", 5, 70, 6, 15),   # Y better, X much wordier -> verbosity wins
    Pair("p12", 6, 35, 5, 25),   # X clearly better
]


def answer_score(quality, words, in_slot_a):
    """The mock judge's biased score for one answer in one slot."""
    bonus = POS_BIAS if in_slot_a else 0.0
    return quality + bonus + VERB_PER_WORD * words


def judge(slot_a, slot_b):
    """Given (quality, words) for the answer in slot A and in slot B,
    return 'A' or 'B' - the winning SLOT. Ties go to A (position bias again)."""
    sa = answer_score(slot_a[0], slot_a[1], in_slot_a=True)
    sb = answer_score(slot_b[0], slot_b[1], in_slot_a=False)
    return "A" if sa >= sb else "B"


def verdict(pair, x_first):
    """Winning SYSTEM ('X' or 'Y') when X is (or is not) shown first in slot A."""
    x = (pair.xq, pair.xw)
    y = (pair.yq, pair.yw)
    if x_first:
        return "X" if judge(x, y) == "A" else "Y"
    else:
        return "Y" if judge(y, x) == "A" else "X"


def true_winner(pair):
    """What a fair judge would say: the higher-quality answer wins."""
    return "X" if pair.xq > pair.yq else "Y"


def corrected_winner(pair):
    """Correction: run BOTH orderings and average each system's two scores.
    Every answer sits in slot A exactly once, so the position bonus cancels."""
    x_avg = (answer_score(pair.xq, pair.xw, True)
             + answer_score(pair.xq, pair.xw, False)) / 2
    y_avg = (answer_score(pair.yq, pair.yw, True)
             + answer_score(pair.yq, pair.yw, False)) / 2
    return "X" if x_avg >= y_avg else "Y"


def rate(count, total):
    return count / total if total else 0.0


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 5 - JUDGING THE JUDGE: measuring a biased judge, then fixing it.")
    print(line)
    print(f"Judge biases baked in:  slot-A bonus = +{POS_BIAS:.2f} quality-points,"
          f"  +{VERB_PER_WORD:.2f}/word")
    print(f"Paired comparisons:     {len(PAIRS)} prompts, system X vs system Y")

    # --- 1. Position bias: swap A/B and count the flips -------------------
    print("\n" + "-" * 72)
    print("1) POSITION BIAS - swap who goes first, count the verdicts that flip.")
    print("-" * 72)
    header = f"{'pair':>4}  {'X(q,words)':>11}  {'Y(q,words)':>11}  {'X-1st':>5}  {'Y-1st':>5}  {'flip':>4}  {'true':>4}"
    print(header)
    flips = 0
    naive_wrong = 0
    for p in PAIRS:
        x_first = verdict(p, x_first=True)     # naive: X always shown first
        y_first = verdict(p, x_first=False)    # the swapped ordering
        flipped = x_first != y_first
        flips += flipped
        naive_wrong += (x_first != true_winner(p))
        print(f"{p.id:>4}  {f'({p.xq},{p.xw})':>11}  {f'({p.yq},{p.yw})':>11}"
              f"  {x_first:>5}  {y_first:>5}  {('YES' if flipped else '.'):>4}"
              f"  {true_winner(p):>4}")
    pos_rate = rate(flips, len(PAIRS))
    print(f"\n  position-bias rate = flips / pairs = {flips}/{len(PAIRS)} = {pos_rate:.2f}")
    print(f"  a fair judge would flip 0 times; ours flips on every close call.")

    # --- 2. Verbosity bias: how often does the longer answer win? ---------
    # Isolate length from position by using the corrected (order-averaged)
    # verdict, so the ONLY thing left to sway a tie in quality is word count.
    print("\n" + "-" * 72)
    print("2) VERBOSITY BIAS - equal-quality answers, does the longer one win?")
    print("-" * 72)
    print("   Order-averaged verdicts on 5 pairs whose two answers are TIED in")
    print("   quality and differ only in length:")
    probes = [
        Pair("q01", 5, 50, 5, 20),   # X longer
        Pair("q02", 6, 20, 6, 45),   # Y longer
        Pair("q03", 4, 55, 4, 25),   # X longer
        Pair("q04", 7, 30, 7, 60),   # Y longer
        Pair("q05", 3, 40, 3, 15),   # X longer
    ]
    print(f"   {'pair':>4}  {'X words':>7}  {'Y words':>7}  {'longer':>6}  {'winner':>6}")
    longer_wins = 0
    for p in probes:
        w = corrected_winner(p)
        longer = "X" if p.xw > p.yw else "Y"
        longer_wins += (w == longer)
        print(f"   {p.id:>4}  {p.xw:>7}  {p.yw:>7}  {longer:>6}  {w:>6}")
    verb_rate = rate(longer_wins, len(probes))
    print(f"\n  verbosity-bias rate = longer-wins / pairs = {longer_wins}/{len(probes)} = {verb_rate:.2f}")
    print(f"  quality is tied, yet the longer answer wins every time.")

    # --- 3. The correction: average both orderings ------------------------
    print("\n" + "-" * 72)
    print("3) CORRECTION - run both orderings and average; the slot-A bonus")
    print("   cancels (each answer sits in slot A once). Compare to the truth.")
    print("-" * 72)
    corr_wrong = 0
    corr_flips = 0  # a "flip" needs the verdict to depend on order; it no longer does
    for p in PAIRS:
        c = corrected_winner(p)
        corr_wrong += (c != true_winner(p))
    naive_rate = rate(naive_wrong, len(PAIRS))
    corr_rate = rate(corr_wrong, len(PAIRS))
    print(f"  naive judge (X always first):  disagreement with truth"
          f" = {naive_wrong}/{len(PAIRS)} = {naive_rate:.2f}")
    print(f"  corrected (both orders avg):   disagreement with truth"
          f" = {corr_wrong}/{len(PAIRS)} = {corr_rate:.2f}")
    print(f"  corrected position-bias rate:  {corr_flips}/{len(PAIRS)} = "
          f"{rate(corr_flips, len(PAIRS)):.2f}  (order can no longer flip a verdict)")
    print(f"\n  -> Averaging killed the position bias, cutting disagreement from")
    print(f"     {naive_wrong}/{len(PAIRS)} to {corr_wrong}/{len(PAIRS)}. The lone survivor is a"
          f" VERBOSITY error:")
    print(f"     order-averaging fixes WHERE an answer sits, not HOW LONG it is.")
    print(line)

# Expected output:
# ========================================================================
# PART 5 - JUDGING THE JUDGE: measuring a biased judge, then fixing it.
# ========================================================================
# Judge biases baked in:  slot-A bonus = +1.00 quality-points,  +0.02/word
# Paired comparisons:     12 prompts, system X vs system Y
#
# ------------------------------------------------------------------------
# 1) POSITION BIAS - swap who goes first, count the verdicts that flip.
# ------------------------------------------------------------------------
# pair   X(q,words)   Y(q,words)  X-1st  Y-1st  flip  true
#  p01       (5,40)       (6,20)      X      Y   YES     Y
#  p02       (6,50)       (7,25)      X      Y   YES     Y
#  p03       (3,45)       (4,30)      X      Y   YES     Y
#  p04       (7,35)       (8,25)      X      Y   YES     Y
#  p05       (9,30)       (4,40)      X      X     .     X
#  p06       (8,20)       (5,30)      X      X     .     X
#  p07       (3,30)       (8,20)      Y      Y     .     Y
#  p08       (4,25)       (7,35)      Y      Y     .     Y
#  p09       (7,25)       (6,40)      X      Y   YES     X
#  p10       (8,20)       (7,45)      X      Y   YES     X
#  p11       (5,70)       (6,15)      X      Y   YES     Y
#  p12       (6,35)       (5,25)      X      X     .     X
#
#   position-bias rate = flips / pairs = 7/12 = 0.58
#   a fair judge would flip 0 times; ours flips on every close call.
#
# ------------------------------------------------------------------------
# 2) VERBOSITY BIAS - equal-quality answers, does the longer one win?
# ------------------------------------------------------------------------
#    Order-averaged verdicts on 5 pairs whose two answers are TIED in
#    quality and differ only in length:
#    pair  X words  Y words  longer  winner
#     q01       50       20       X       X
#     q02       20       45       Y       Y
#     q03       55       25       X       X
#     q04       30       60       Y       Y
#     q05       40       15       X       X
#
#   verbosity-bias rate = longer-wins / pairs = 5/5 = 1.00
#   quality is tied, yet the longer answer wins every time.
#
# ------------------------------------------------------------------------
# 3) CORRECTION - run both orderings and average; the slot-A bonus
#    cancels (each answer sits in slot A once). Compare to the truth.
# ------------------------------------------------------------------------
#   naive judge (X always first):  disagreement with truth = 5/12 = 0.42
#   corrected (both orders avg):   disagreement with truth = 1/12 = 0.08
#   corrected position-bias rate:  0/12 = 0.00  (order can no longer flip a verdict)
#
#   -> Averaging killed the position bias, cutting disagreement from
#      5/12 to 1/12. The lone survivor is a VERBOSITY error:
#      order-averaging fixes WHERE an answer sits, not HOW LONG it is.
# ========================================================================
