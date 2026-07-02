"""
Evals from First Principles - Part 14: Agent-Trajectory Evals.

Agents Part 17 ("Grading the Agent") graded whether the agent's FINAL answer
was right. Here we grade the PATH: the sequence of tool calls the agent took
to get there. On a small multi-step transcript versus a reference trajectory
we compute three trajectory-match scores (exact, order-aware, set-overlap),
assign step-level credit (which steps were correct / missing / extra), then
score a pass/fail rubric over eight runs and measure judge-vs-human agreement
with Cohen's kappa (the chance-corrected agreement from Part 3).

Pure Python, offline, deterministic. No LLM, no network: the "judge" is a
transparent rule standing in for an LLM grader so every number is traceable.
"""

# --- The task -------------------------------------------------------------
# An agent must answer: "What does 3 nights at the top-rated Paris hotel cost
# in USD?" A correct run has to (1) search the web, (2) open the hotel page,
# (3) convert the euro price to USD, then (4) answer. That ordered list of
# tool names is the REFERENCE trajectory a human approved.
REFERENCE = ["search", "open", "convert", "answer"]

# Eight actual agent runs, each a sequence of tool calls (the transcript,
# stripped to tool names). Some are clean, some take a messy or lazy path.
RUNS = {
    "R1": ["search", "open", "convert", "answer"],            # textbook path
    "R2": ["open", "search", "convert", "answer"],            # right tools, wrong order
    "R3": ["search", "open", "search", "answer"],            # skipped convert, re-searched
    "R4": ["search", "open", "convert", "convert", "answer"],  # converted twice
    "R5": ["search", "answer"],                              # lazy: skipped open + convert
    "R6": ["search", "open", "convert", "answer", "answer"],  # answered twice
    "R7": ["search", "search", "open", "convert", "answer"],  # extra search up front
    "R8": ["answer"],                                        # guessed, used no tools
}


# --- Trajectory-match scoring --------------------------------------------
def lcs_pairs(cand, ref):
    """Longest common subsequence of two tool sequences, returned as the list
    of matched (cand_index, ref_index) pairs. This is the by-hand DP table
    from a diff: it finds the longest in-order overlap, tolerating extra or
    missing steps on either side."""
    n, m = len(cand), len(ref)
    # table[i][j] = LCS length of cand[i:] and ref[j:]
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if cand[i] == ref[j]:
                table[i][j] = 1 + table[i + 1][j + 1]
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    # Walk the table forward to recover which steps matched.
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if cand[i] == ref[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def exact_match(cand, ref):
    """1.0 only if the trajectory is identical to the reference, else 0.0."""
    return 1.0 if cand == ref else 0.0


def order_aware(cand, ref):
    """How much of the reference path we recovered IN ORDER: LCS / len(ref)."""
    return len(lcs_pairs(cand, ref)) / len(ref) if ref else 0.0


def set_overlap(cand, ref):
    """Order-free tool overlap: Jaccard of the two tool SETS (dedups steps)."""
    a, b = set(cand), set(ref)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def step_credit(cand, ref):
    """Align cand to ref and label every step. Returns (correct, missing,
    extra, credit) where credit = correct / (correct + missing + extra)."""
    pairs = lcs_pairs(cand, ref)
    correct = len(pairs)
    missing = len(ref) - correct   # reference steps never matched
    extra = len(cand) - correct    # candidate steps that matched nothing
    denom = correct + missing + extra
    credit = correct / denom if denom else 0.0
    return correct, missing, extra, credit


def labeled_steps(cand, ref):
    """Per-step labels for one candidate: each step is MATCH or EXTRA, plus
    the list of reference steps that were MISSING."""
    matched_cand = {i for i, _ in lcs_pairs(cand, ref)}
    matched_ref = {j for _, j in lcs_pairs(cand, ref)}
    labels = [(t, "MATCH" if i in matched_cand else "EXTRA")
              for i, t in enumerate(cand)]
    missing = [t for j, t in enumerate(ref) if j not in matched_ref]
    return labels, missing


# --- The rubric: a mock judge vs a human ---------------------------------
# Human gold rubric: PASS if the run took an acceptable path (hit every needed
# step in a sensible order without a wrong-order or skipped-step mistake; a
# harmless duplicate step is tolerated). These are the labels a human wrote.
HUMAN = {"R1": "PASS", "R2": "FAIL", "R3": "FAIL", "R4": "PASS",
         "R5": "FAIL", "R6": "PASS", "R7": "PASS", "R8": "FAIL"}


def judge(cand, ref):
    """Deterministic stand-in for an LLM judge grading the same rubric. Its
    (deliberately imperfect) rule: PASS if it recovered the whole reference
    path in order AND added at most one extra step. It never sees that a
    wrong ORDER can still yield a high partial score, so it is too lenient."""
    _, _, extra, _ = step_credit(cand, ref)
    return "PASS" if (order_aware(cand, ref) >= 0.75 and extra <= 1) else "FAIL"


def cohen_kappa(rater_a, rater_b, keys):
    """Cohen's kappa on two PASS/FAIL raters over the same items (from Part 3).
    Builds the 2x2, then kappa = (po - pe) / (1 - pe)."""
    a = sum(rater_a[k] == "PASS" and rater_b[k] == "PASS" for k in keys)  # both PASS
    d = sum(rater_a[k] == "FAIL" and rater_b[k] == "FAIL" for k in keys)  # both FAIL
    b = sum(rater_a[k] == "PASS" and rater_b[k] == "FAIL" for k in keys)
    c = sum(rater_a[k] == "FAIL" and rater_b[k] == "PASS" for k in keys)
    n = a + b + c + d
    po = (a + d) / n
    # chance agreement from the marginals
    pe = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return a, b, c, d, po, pe, kappa


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 14 - AGENT-TRAJECTORY EVALS: grade the path, not just the answer.")
    print(line)
    print("Task: find the 3-night cost of the top Paris hotel in USD.")
    print("Reference trajectory (the path a human approved):")
    print("  " + " > ".join(f"{i}.{t}" for i, t in enumerate(REFERENCE, 1)))

    # --- 1. Three trajectory-match variants on one run --------------------
    print("\n" + "-" * 72)
    print("1. THREE WAYS TO MATCH A TRAJECTORY (run R2: right tools, wrong order)")
    print("-" * 72)
    cand = RUNS["R2"]
    print("  candidate : " + " > ".join(cand))
    print("  reference : " + " > ".join(REFERENCE))
    nlcs = len(lcs_pairs(cand, REFERENCE))
    inter = len(set(cand) & set(REFERENCE))
    union = len(set(cand) | set(REFERENCE))
    print(f"  exact-match  : cand == ref ?           -> {exact_match(cand, REFERENCE):.2f}")
    print(f"  order-aware  : LCS={nlcs} / len(ref)={len(REFERENCE)}       -> {order_aware(cand, REFERENCE):.2f}")
    print(f"  set-overlap  : |cap|={inter} / |cup|={union}          -> {set_overlap(cand, REFERENCE):.2f}")
    print("  Same four tools (set says 1.00), but out of order (order says 0.75),")
    print("  and not identical (exact says 0.00). The metric you pick is the verdict.")

    # --- 2. Step-level credit assignment ----------------------------------
    print("\n" + "-" * 72)
    print("2. STEP-LEVEL CREDIT (run R3: skipped convert, re-searched instead)")
    print("-" * 72)
    cand = RUNS["R3"]
    labels, missing = labeled_steps(cand, REFERENCE)
    for i, (t, lab) in enumerate(labels, 1):
        print(f"  step {i}: {t:<8} -> {lab}")
    for t in missing:
        print(f"  (ref)   {t:<8} -> MISSING")
    correct, miss, extra, credit = step_credit(cand, REFERENCE)
    print(f"  credit = correct/(correct+missing+extra) = "
          f"{correct}/({correct}+{miss}+{extra}) = {credit:.2f}")

    # --- 3. All eight runs scored -----------------------------------------
    print("\n" + "-" * 72)
    print("3. ALL EIGHT AGENT RUNS, SCORED FOUR WAYS")
    print("-" * 72)
    print(f"  {'run':<4}{'trajectory':<34}{'exact':>6}{'order':>7}{'set':>6}{'credit':>8}")
    for k, traj in RUNS.items():
        ex = exact_match(traj, REFERENCE)
        od = order_aware(traj, REFERENCE)
        so = set_overlap(traj, REFERENCE)
        _, _, _, cr = step_credit(traj, REFERENCE)
        print(f"  {k:<4}{'>'.join(traj):<34}{ex:>6.2f}{od:>7.2f}{so:>6.2f}{cr:>8.2f}")

    # --- 4. Rubric: judge vs human, and kappa -----------------------------
    print("\n" + "-" * 72)
    print("4. RUBRIC GRADE: MOCK JUDGE vs HUMAN, WITH COHEN'S KAPPA")
    print("-" * 72)
    JUDGE = {k: judge(traj, REFERENCE) for k, traj in RUNS.items()}
    print(f"  {'run':<4}{'human':>7}{'judge':>7}   {'agree?':>7}")
    for k in RUNS:
        agree = "yes" if HUMAN[k] == JUDGE[k] else "NO"
        print(f"  {k:<4}{HUMAN[k]:>7}{JUDGE[k]:>7}   {agree:>7}")

    a, b, c, d, po, pe, kappa = cohen_kappa(JUDGE, HUMAN, list(RUNS))
    print(f"\n  2x2 (judge rows, human cols):")
    print(f"                human=PASS  human=FAIL")
    print(f"    judge=PASS      {a}           {b}")
    print(f"    judge=FAIL      {c}           {d}")
    print(f"  observed agreement po = (a+d)/n = ({a}+{d})/8 = {po:.2f}")
    print(f"  chance agreement   pe            = {pe:.2f}")
    print(f"  Cohen's kappa = (po-pe)/(1-pe) = ({po:.2f}-{pe:.2f})/(1-{pe:.2f}) = {kappa:.2f}")
    print("  Judge and human agree 75% of the time, but both mostly say PASS, so")
    print("  chance alone would hit 0.50; kappa=0.50 is only 'moderate'. The two")
    print("  misses (R2, R3) are exactly the wrong-order / skipped-step paths the")
    print("  lenient judge waved through. Grading the trajectory caught them.")
    print(line)

# Expected output:
# ========================================================================
# PART 14 - AGENT-TRAJECTORY EVALS: grade the path, not just the answer.
# ========================================================================
# Task: find the 3-night cost of the top Paris hotel in USD.
# Reference trajectory (the path a human approved):
#   1.search > 2.open > 3.convert > 4.answer
#
# ------------------------------------------------------------------------
# 1. THREE WAYS TO MATCH A TRAJECTORY (run R2: right tools, wrong order)
# ------------------------------------------------------------------------
#   candidate : open > search > convert > answer
#   reference : search > open > convert > answer
#   exact-match  : cand == ref ?           -> 0.00
#   order-aware  : LCS=3 / len(ref)=4       -> 0.75
#   set-overlap  : |cap|=4 / |cup|=4          -> 1.00
#   Same four tools (set says 1.00), but out of order (order says 0.75),
#   and not identical (exact says 0.00). The metric you pick is the verdict.
#
# ------------------------------------------------------------------------
# 2. STEP-LEVEL CREDIT (run R3: skipped convert, re-searched instead)
# ------------------------------------------------------------------------
#   step 1: search   -> MATCH
#   step 2: open     -> MATCH
#   step 3: search   -> EXTRA
#   step 4: answer   -> MATCH
#   (ref)   convert  -> MISSING
#   credit = correct/(correct+missing+extra) = 3/(3+1+1) = 0.60
#
# ------------------------------------------------------------------------
# 3. ALL EIGHT AGENT RUNS, SCORED FOUR WAYS
# ------------------------------------------------------------------------
#   run trajectory                         exact  order   set  credit
#   R1  search>open>convert>answer          1.00   1.00  1.00    1.00
#   R2  open>search>convert>answer          0.00   0.75  1.00    0.60
#   R3  search>open>search>answer           0.00   0.75  0.75    0.60
#   R4  search>open>convert>convert>answer  0.00   1.00  1.00    0.80
#   R5  search>answer                       0.00   0.50  0.50    0.50
#   R6  search>open>convert>answer>answer   0.00   1.00  1.00    0.80
#   R7  search>search>open>convert>answer   0.00   1.00  1.00    0.80
#   R8  answer                              0.00   0.25  0.25    0.25
#
# ------------------------------------------------------------------------
# 4. RUBRIC GRADE: MOCK JUDGE vs HUMAN, WITH COHEN'S KAPPA
# ------------------------------------------------------------------------
#   run   human  judge    agree?
#   R1     PASS   PASS       yes
#   R2     FAIL   PASS        NO
#   R3     FAIL   PASS        NO
#   R4     PASS   PASS       yes
#   R5     FAIL   FAIL       yes
#   R6     PASS   PASS       yes
#   R7     PASS   PASS       yes
#   R8     FAIL   FAIL       yes
#
#   2x2 (judge rows, human cols):
#                 human=PASS  human=FAIL
#     judge=PASS      4           2
#     judge=FAIL      0           2
#   observed agreement po = (a+d)/n = (4+2)/8 = 0.75
#   chance agreement   pe            = 0.50
#   Cohen's kappa = (po-pe)/(1-pe) = (0.75-0.50)/(1-0.50) = 0.50
#   Judge and human agree 75% of the time, but both mostly say PASS, so
#   chance alone would hit 0.50; kappa=0.50 is only 'moderate'. The two
#   misses (R2, R3) are exactly the wrong-order / skipped-step paths the
#   lenient judge waved through. Grading the trajectory caught them.
# ========================================================================
