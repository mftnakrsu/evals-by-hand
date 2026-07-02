"""
Evals from First Principles - Part 4: LLM-as-Judge by Hand.

Part 3 measured how much two humans disagree. Here we replace (or scale) the
human grader with a MODEL and ask the same question of the machine: how often
does the judge agree with the human gold, and exactly where does it break?

We build a rubric prompt, run a DETERMINISTIC mock judge over a small labeled
set, PARSE its free-text verdict into PASS / FAIL / ABSTAIN, and score the
judge against the human gold with the same 2x2 confusion matrix from Part 1.
Along the way we count the three failure modes that wreck real judges: a tie
(the judge will not commit), a refusal (the candidate declined to answer), and
format drift (the verdict comes back with no parseable tag). Pure Python,
offline, deterministic - no network, no API key, no real model.
"""

# --- The rubric prompt ----------------------------------------------------
# This is the instruction we would send a real LLM judge. The mock judge below
# obeys exactly these rules, deterministically, so every verdict is traceable.
RUBRIC = """You are grading one answer. Reply on ONE line as: VERDICT: PASS
or VERDICT: FAIL.
  - FAIL if the answer refuses or declines to respond.
  - PASS only if the answer contains the required key fact.
  - Otherwise FAIL."""


# --- The labeled set ------------------------------------------------------
# Each item: a question, the KEY fact a good answer must contain, the CANDIDATE
# answer under test, the HUMAN gold grade (1 = a human said PASS, 0 = FAIL),
# and a human-written NOTE that names why the item is interesting (display
# only; it is never used in any computation).
ITEMS = [
    dict(qid="Q01", key="paris",    human=1,
         candidate="The capital of France is Paris.",
         note="clean pass"),
    dict(qid="Q02", key="tokyo",    human=1,
         candidate="Tokyo.",
         note="correct but terse -> TIE"),
    dict(qid="Q03", key="ignition", human=0,
         candidate="I cannot help with that request.",
         note="refusal"),
    dict(qid="Q04", key="jupiter",  human=1,
         candidate="The largest planet is Jupiter.",
         note="clean pass"),
    dict(qid="Q05", key="canberra", human=0,
         candidate="Sydney is the capital, though Canberra is the seat of parliament.",
         note="wrong, but the key word appears -> keyword-gamed"),
    dict(qid="Q06", key="washington", human=1,
         candidate="The answer is D.C.",
         note="correct via synonym the judge misses"),
    dict(qid="Q07", key="sphere",   human=1,
         candidate=("Well, if you really think about it in great detail, our "
                    "planet is not perfectly flat at all, and in fact people "
                    "have shown for many centuries that the earth is basically "
                    "a slightly squashed sphere."),
         note="verbose -> format drift"),
    dict(qid="Q08", key="pacific",  human=1,
         candidate="The Pacific Ocean is the largest.",
         note="clean pass"),
    dict(qid="Q09", key="five",     human=1,
         candidate="The result is 5.",
         note="correct as a digit the judge misses"),
    dict(qid="Q10", key="h2o",      human=1,
         candidate="Water is H2O.",
         note="clean pass"),
    dict(qid="Q11", key="brasilia", human=0,
         candidate="The capital of Brazil is Rio de Janeiro.",
         note="wrong answer, correctly failed"),
    dict(qid="Q12", key="four",     human=0,
         candidate="The answer is five.",
         note="wrong answer, correctly failed"),
]

REFUSAL_CUES = ("cannot", "can't", "unable", "i won't", "i will not")


def is_refusal(text):
    """True if the candidate declined to answer."""
    low = text.lower()
    return any(cue in low for cue in REFUSAL_CUES)


def mock_judge(candidate, key):
    """A deterministic stand-in for an LLM judge.

    It returns the RAW verdict STRING a model would emit, including the messy
    cases: a tie when there is too little to grade, and format drift (no
    VERDICT tag) when the answer is long enough to make the model ramble.
    """
    words = candidate.split()
    if is_refusal(candidate):
        return "VERDICT: FAIL  (the answer refuses to respond)"
    if len(words) < 3:
        return "VERDICT: TIE  (too little content to grade)"
    if len(words) > 25:
        # A long answer makes the model ramble and forget the format.
        return "After weighing both sides, this answer looks essentially fine to me."
    if key in candidate.lower():
        return "VERDICT: PASS"
    return "VERDICT: FAIL"


def parse_verdict(raw):
    """Turn the judge's free text into 1 (PASS), 0 (FAIL), or None (ABSTAIN).

    ABSTAIN covers everything we cannot trust: a TIE token, or any reply with
    no 'VERDICT:' tag at all (format drift).
    """
    marker = "VERDICT:"
    if marker not in raw:
        return None
    token = raw.split(marker, 1)[1].strip().split()[0].strip(".").upper()
    if token == "PASS":
        return 1
    if token == "FAIL":
        return 0
    return None


def confusion(gold, pred):
    """The four cells of the 2x2, positive class = PASS. From Part 1."""
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def grade(word):
    return "PASS" if word == 1 else "FAIL"


def shorten(raw, width=42):
    return raw if len(raw) <= width else raw[:width].rstrip() + "..."


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 4 - LLM-AS-JUDGE BY HAND: scoring the judge against human gold.")
    print(line)
    print("The rubric we hand the judge:")
    for r in RUBRIC.splitlines():
        print("  | " + r)

    # --- Run the judge over every item, parsing each verdict ---------------
    print("\n" + "-" * 72)
    print("Judge verdicts (raw model text -> parsed label):")
    print("-" * 72)
    gold_scored, judge_scored = [], []
    refusals = ties = drift = 0
    for it in ITEMS:
        raw = mock_judge(it["candidate"], it["key"])
        parsed = parse_verdict(raw)
        if is_refusal(it["candidate"]):
            refusals += 1
        if parsed is None:
            if "VERDICT:" not in raw:
                drift += 1
            else:
                ties += 1
            label = "ABSTAIN"
        else:
            label = grade(parsed)
            gold_scored.append(it["human"])
            judge_scored.append(parsed)
        print(f"  {it['qid']}  gold={grade(it['human'])}  judge->{label:<8}"
              f"  raw: {shorten(raw)}")

    # --- Count the failure modes ------------------------------------------
    scored = len(gold_scored)
    abstained = len(ITEMS) - scored
    print("\n" + "-" * 72)
    print("FAILURE MODES the parser had to survive:")
    print("-" * 72)
    print(f"  refusals (candidate declined) : {refusals}"
          "   -> judge said FAIL, kept in the matrix")
    print(f"  ties     (judge won't commit) : {ties}"
          "   -> ABSTAIN, dropped from the matrix")
    print(f"  format drift (no VERDICT tag) : {drift}"
          "   -> ABSTAIN, dropped from the matrix")
    print(f"  scorable by the judge         : {scored} of {len(ITEMS)}"
          f"   ({abstained} abstained)")

    # --- Judge vs human: the confusion matrix and metrics -----------------
    tp, fp, fn, tn = confusion(gold_scored, judge_scored)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print("\n" + "-" * 72)
    print("JUDGE vs HUMAN on the scorable set (positive = PASS):")
    print("-" * 72)
    print(f"  n = {n}  (human PASS: {tp + fn}, human FAIL: {fp + tn})")
    print(f"  confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  accuracy  = (TP+TN)/n  = ({tp}+{tn})/{n} = {acc:.2f}")
    print(f"  precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {prec:.2f}")
    print(f"  recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {rec:.2f}")
    print(f"  F1        = 2PR/(P+R)  = {f1:.2f}")
    print()
    print("  The one FP (Q05) is keyword-gaming: a wrong answer that happened to")
    print("  contain the key word. The two FN (Q06, Q09) are correct answers the")
    print("  judge failed for using a synonym or a digit. A judge is just another")
    print("  noisy grader - measure it against humans before you trust its score.")
    print(line)

# Expected output:
# ========================================================================
# PART 4 - LLM-AS-JUDGE BY HAND: scoring the judge against human gold.
# ========================================================================
# The rubric we hand the judge:
#   | You are grading one answer. Reply on ONE line as: VERDICT: PASS
#   | or VERDICT: FAIL.
#   |   - FAIL if the answer refuses or declines to respond.
#   |   - PASS only if the answer contains the required key fact.
#   |   - Otherwise FAIL.
#
# ------------------------------------------------------------------------
# Judge verdicts (raw model text -> parsed label):
# ------------------------------------------------------------------------
#   Q01  gold=PASS  judge->PASS      raw: VERDICT: PASS
#   Q02  gold=PASS  judge->ABSTAIN   raw: VERDICT: TIE  (too little content to grade...
#   Q03  gold=FAIL  judge->FAIL      raw: VERDICT: FAIL  (the answer refuses to resp...
#   Q04  gold=PASS  judge->PASS      raw: VERDICT: PASS
#   Q05  gold=FAIL  judge->PASS      raw: VERDICT: PASS
#   Q06  gold=PASS  judge->FAIL      raw: VERDICT: FAIL
#   Q07  gold=PASS  judge->ABSTAIN   raw: After weighing both sides, this answer loo...
#   Q08  gold=PASS  judge->PASS      raw: VERDICT: PASS
#   Q09  gold=PASS  judge->FAIL      raw: VERDICT: FAIL
#   Q10  gold=PASS  judge->PASS      raw: VERDICT: PASS
#   Q11  gold=FAIL  judge->FAIL      raw: VERDICT: FAIL
#   Q12  gold=FAIL  judge->FAIL      raw: VERDICT: FAIL
#
# ------------------------------------------------------------------------
# FAILURE MODES the parser had to survive:
# ------------------------------------------------------------------------
#   refusals (candidate declined) : 1   -> judge said FAIL, kept in the matrix
#   ties     (judge won't commit) : 1   -> ABSTAIN, dropped from the matrix
#   format drift (no VERDICT tag) : 1   -> ABSTAIN, dropped from the matrix
#   scorable by the judge         : 10 of 12   (2 abstained)
#
# ------------------------------------------------------------------------
# JUDGE vs HUMAN on the scorable set (positive = PASS):
# ------------------------------------------------------------------------
#   n = 10  (human PASS: 6, human FAIL: 4)
#   confusion:  TP=4  FP=1  FN=2  TN=3
#   accuracy  = (TP+TN)/n  = (4+3)/10 = 0.70
#   precision = TP/(TP+FP) = 4/(4+1) = 0.80
#   recall    = TP/(TP+FN) = 4/(4+2) = 0.67
#   F1        = 2PR/(P+R)  = 0.73
#
#   The one FP (Q05) is keyword-gaming: a wrong answer that happened to
#   contain the key word. The two FN (Q06, Q09) are correct answers the
#   judge failed for using a synonym or a digit. A judge is just another
#   noisy grader - measure it against humans before you trust its score.
# ========================================================================
