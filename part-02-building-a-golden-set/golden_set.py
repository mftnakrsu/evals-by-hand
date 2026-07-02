"""
Evals from First Principles — Part 2: Building a Golden Set.

A metric is only as good as the labeled set under it. This part builds a
trustworthy eval set by hand on tiny, printable data:

  1. STRATIFIED vs SIMPLE-RANDOM sampling on a 40-item population with a rare
     class (8 urgent, 32 not). Stratification fixes the rare-class share, so it
     controls variance; simple random sampling can miss the rare class entirely.
  2. The LEAKAGE trap: a few eval items are near-duplicates of "train" items a
     model has memorized. They inflate measured accuracy; removing them exposes
     the honest, lower number.
  3. A 3-level RUBRIC that turns a vague "good answer" into countable criteria.

Pure Python + NumPy, offline, deterministic. All randomness uses a fixed seed
(numpy.random.default_rng(0)), so every number below is reproducible. The
hypergeometric facts are computed exactly with math.comb and cross-checked
against the simulation, never the other way around.
"""

import math
import numpy as np

SEED = 0
BAR = "=" * 72
SUB = "-" * 72


# --- Section 1: stratified vs simple-random sampling ----------------------
# The population: 40 support tickets. The class we care about ("urgent") is
# RARE: 8 urgent (label 1) and 32 not-urgent (label 0). We can only afford to
# hand-label a SAMPLE of n=10 for our golden set. How we draw it matters.
POP_URGENT = 8
POP_NOT = 32
POP_N = POP_URGENT + POP_NOT          # 40
POP_RATE = POP_URGENT / POP_N         # 0.20 true urgent rate
SAMPLE_N = 10


def make_population():
    """Return the 40 labels: 8 ones (urgent) then 32 zeros (not)."""
    return np.array([1] * POP_URGENT + [0] * POP_NOT)


def simple_random_sample(pop, n, rng):
    """Draw n items uniformly WITHOUT replacement from the whole population."""
    idx = rng.choice(len(pop), size=n, replace=False)
    return pop[idx]


def stratified_sample(pop, n, rng):
    """Draw n items keeping each class's share fixed at its population rate.

    urgent share = 0.20, so a sample of 10 gets exactly 2 urgent + 8 not,
    drawn without replacement WITHIN each stratum.
    """
    urgent_idx = np.where(pop == 1)[0]
    not_idx = np.where(pop == 0)[0]
    take_urgent = round(n * POP_URGENT / len(pop))   # 10 * 8/40 = 2
    take_not = n - take_urgent                        # 8
    pick_u = rng.choice(urgent_idx, size=take_urgent, replace=False)
    pick_n = rng.choice(not_idx, size=take_not, replace=False)
    return pop[np.concatenate([pick_u, pick_n])]


def prob_zero_urgent_exact():
    """Exact P(a simple random sample of 10 contains ZERO urgent tickets).

    Hypergeometric: choose all 10 from the 32 not-urgent, over all C(40,10).
    """
    return math.comb(POP_NOT, SAMPLE_N) / math.comb(POP_N, SAMPLE_N)


def std_urgent_count_exact():
    """Exact std of the urgent COUNT in a simple random sample of 10.

    Hypergeometric variance: n * p * (1-p) * (N-n)/(N-1).
    """
    p = POP_RATE
    var = SAMPLE_N * p * (1 - p) * (POP_N - SAMPLE_N) / (POP_N - 1)
    return math.sqrt(var)


def section_sampling():
    print(BAR)
    print("PART 2 - BUILDING A GOLDEN SET")
    print(BAR)
    print("Population: 40 tickets, 8 urgent (rare) + 32 not. True urgent rate = 0.20.")
    print(f"We can label only n={SAMPLE_N} for the golden set.\n")

    rng = np.random.default_rng(SEED)
    pop = make_population()

    # One draw of each kind, to see them concretely.
    srs = simple_random_sample(pop, SAMPLE_N, rng)
    strat = stratified_sample(pop, SAMPLE_N, rng)
    print("One SIMPLE-RANDOM sample of 10:")
    print(f"  urgent picked = {int(srs.sum())}  ->  measured urgent rate = {srs.mean():.2f}")
    print("One STRATIFIED sample of 10 (share fixed at 0.20):")
    print(f"  urgent picked = {int(strat.sum())}  ->  measured urgent rate = {strat.mean():.2f}")

    # Repeat many times to compare the SPREAD of the measured rate.
    trials = 10000
    rng = np.random.default_rng(SEED)
    srs_counts = np.array([simple_random_sample(pop, SAMPLE_N, rng).sum()
                           for _ in range(trials)])
    strat_counts = np.array([stratified_sample(pop, SAMPLE_N, rng).sum()
                             for _ in range(trials)])
    srs_zero = int((srs_counts == 0).sum())

    print(f"\nOver {trials} resamples (fixed seed):")
    print(f"  simple-random : mean urgent = {srs_counts.mean():.3f}, "
          f"std = {srs_counts.std():.3f}")
    print(f"  stratified    : mean urgent = {strat_counts.mean():.3f}, "
          f"std = {strat_counts.std():.3f}")
    print(f"  simple-random samples with ZERO urgent: "
          f"{srs_zero}/{trials} = {srs_zero / trials:.3f}")

    print("\nExact (hypergeometric) cross-check of the simple-random draw:")
    print(f"  P(zero urgent)      = C(32,10)/C(40,10) = {prob_zero_urgent_exact():.3f}")
    print(f"  std of urgent count = sqrt(n*p*(1-p)*(N-n)/(N-1)) = "
          f"{std_urgent_count_exact():.3f}")
    print("  -> stratifying pins the rare-class share, driving its std to 0; simple")
    print("     random sampling has real spread and can miss urgent tickets entirely.")


# --- Section 2: the leakage trap ------------------------------------------
# A model was "trained" on TRAIN_TICKETS. Our eval set EVAL_TICKETS shares a few
# near-duplicates with it (same text up to case / punctuation / spacing). The
# model has memorized those, so it answers them for free. Counting them inflates
# accuracy; a golden set must not overlap the training data.
TRAIN_TICKETS = [
    "How do I reset my password?",
    "My invoice is wrong",
    "The app crashes on login",
    "Where is my refund?",
    "Cancel my subscription",
    "Update my billing address",
]

# (eval text, model got it right?) — memorized duplicates are always "right".
EVAL_TICKETS = [
    ("how do i reset my password", 1),   # near-dup of train #1 (memorized)
    ("The app crashes on login.", 1),     # near-dup of train #3 (memorized)
    ("How do I export my data?", 1),      # genuine, correct
    ("Is there a mobile app?", 1),        # genuine, correct
    ("What are your business hours?", 0),  # genuine, wrong
    ("cancel my subscription!", 1),        # near-dup of train #5 (memorized)
    ("How do I change my email?", 1),      # genuine, correct
    ("Do you offer discounts?", 0),        # genuine, wrong
    ("My invoice is wrong.", 1),           # near-dup of train #2 (memorized)
    ("Can I get a receipt?", 1),           # genuine, correct
]


def normalize(text):
    """Lowercase, keep only alphanumerics and spaces, collapse whitespace."""
    kept = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return " ".join(kept.split())


def find_leaks(train, eval_set):
    """Return the eval indices whose normalized text also appears in train."""
    train_norm = {normalize(t) for t in train}
    return [i for i, (text, _) in enumerate(eval_set)
            if normalize(text) in train_norm]


def accuracy(rows):
    return sum(correct for _, correct in rows) / len(rows) if rows else 0.0


def section_leakage():
    print("\n" + SUB)
    print("LEAKAGE: near-duplicates of training data sneak into the eval set.")
    print(SUB)
    leaks = find_leaks(TRAIN_TICKETS, EVAL_TICKETS)
    print(f"eval size = {len(EVAL_TICKETS)}   train size = {len(TRAIN_TICKETS)}")
    print(f"leaked eval items (normalized text found in train): {len(leaks)}")
    for i in leaks:
        print(f"  eval[{i}] {EVAL_TICKETS[i][0]!r}  ==  a train ticket")

    with_leak = accuracy(EVAL_TICKETS)
    clean = [row for i, row in enumerate(EVAL_TICKETS) if i not in leaks]
    honest = accuracy(clean)
    leak_correct = sum(EVAL_TICKETS[i][1] for i in leaks)

    print(f"\nmeasured accuracy WITH leakage  = {sum(c for _, c in EVAL_TICKETS)}"
          f"/{len(EVAL_TICKETS)} = {with_leak:.3f}")
    print(f"  (the {len(leaks)} memorized duplicates are all correct: "
          f"{leak_correct}/{len(leaks)})")
    print(f"honest accuracy AFTER removing leaks = "
          f"{sum(c for _, c in clean)}/{len(clean)} = {honest:.3f}")
    print(f"  -> leakage inflated the score by {with_leak - honest:.3f}. The gimme")
    print("     duplicates hid the model's real performance on unseen tickets.")


# --- Section 3: from a vague bar to an operational rubric ------------------
# "Was it a good answer?" is a vibe. A rubric turns it into countable criteria.
# Three criteria, each scored on a 3-level scale: 0 = fails, 1 = partial, 2 = full.
# Max score = 6; an answer PASSES the golden set at total >= 5.
RUBRIC_CRITERIA = ["Correct", "Grounded", "Complete"]
PASS_THRESHOLD = 5

# (label, [Correct, Grounded, Complete]) — three graded candidate answers.
CANDIDATES = [
    ("Answer A", [2, 2, 2]),   # right, grounded, complete
    ("Answer B", [2, 2, 1]),   # right and grounded, but partial
    ("Answer C", [1, 0, 2]),   # complete but half-right and ungrounded
]


def section_rubric():
    print("\n" + SUB)
    print("RUBRIC: turn 'good answer?' into countable criteria (each 0/1/2).")
    print(SUB)
    print(f"criteria: {', '.join(RUBRIC_CRITERIA)}   (max {2 * len(RUBRIC_CRITERIA)}, "
          f"pass at total >= {PASS_THRESHOLD})")
    for label, scores in CANDIDATES:
        total = sum(scores)
        verdict = "PASS" if total >= PASS_THRESHOLD else "FAIL"
        cells = "  ".join(f"{name}={s}" for name, s in zip(RUBRIC_CRITERIA, scores))
        print(f"  {label}: {cells}  ->  total {total}/6  {verdict}")
    passing = sum(1 for _, s in CANDIDATES if sum(s) >= PASS_THRESHOLD)
    print(f"\n{passing}/{len(CANDIDATES)} answers clear the bar. A vibe check might wave")
    print("all three through; the rubric makes the bar a number anyone can re-count.")


if __name__ == "__main__":
    section_sampling()
    section_leakage()
    section_rubric()
    print(BAR)

# Expected output:
# ========================================================================
# PART 2 - BUILDING A GOLDEN SET
# ========================================================================
# Population: 40 tickets, 8 urgent (rare) + 32 not. True urgent rate = 0.20.
# We can label only n=10 for the golden set.
#
# One SIMPLE-RANDOM sample of 10:
#   urgent picked = 4  ->  measured urgent rate = 0.40
# One STRATIFIED sample of 10 (share fixed at 0.20):
#   urgent picked = 2  ->  measured urgent rate = 0.20
#
# Over 10000 resamples (fixed seed):
#   simple-random : mean urgent = 1.997, std = 1.105
#   stratified    : mean urgent = 2.000, std = 0.000
#   simple-random samples with ZERO urgent: 743/10000 = 0.074
#
# Exact (hypergeometric) cross-check of the simple-random draw:
#   P(zero urgent)      = C(32,10)/C(40,10) = 0.076
#   std of urgent count = sqrt(n*p*(1-p)*(N-n)/(N-1)) = 1.109
#   -> stratifying pins the rare-class share, driving its std to 0; simple
#      random sampling has real spread and can miss urgent tickets entirely.
#
# ------------------------------------------------------------------------
# LEAKAGE: near-duplicates of training data sneak into the eval set.
# ------------------------------------------------------------------------
# eval size = 10   train size = 6
# leaked eval items (normalized text found in train): 4
#   eval[0] 'how do i reset my password'  ==  a train ticket
#   eval[1] 'The app crashes on login.'  ==  a train ticket
#   eval[5] 'cancel my subscription!'  ==  a train ticket
#   eval[8] 'My invoice is wrong.'  ==  a train ticket
#
# measured accuracy WITH leakage  = 8/10 = 0.800
#   (the 4 memorized duplicates are all correct: 4/4)
# honest accuracy AFTER removing leaks = 4/6 = 0.667
#   -> leakage inflated the score by 0.133. The gimme
#      duplicates hid the model's real performance on unseen tickets.
#
# ------------------------------------------------------------------------
# RUBRIC: turn 'good answer?' into countable criteria (each 0/1/2).
# ------------------------------------------------------------------------
# criteria: Correct, Grounded, Complete   (max 6, pass at total >= 5)
#   Answer A: Correct=2  Grounded=2  Complete=2  ->  total 6/6  PASS
#   Answer B: Correct=2  Grounded=2  Complete=1  ->  total 5/6  PASS
#   Answer C: Correct=1  Grounded=0  Complete=2  ->  total 3/6  FAIL
#
# 2/3 answers clear the bar. A vibe check might wave
# all three through; the rubric makes the bar a number anyone can re-count.
# ========================================================================
