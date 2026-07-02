"""
Evals from First Principles — Part 1: What Is a Score?

The atoms of every eval: a task, a gold label, and a metric. We build the
2x2 confusion matrix by hand from 10 graded outputs, derive accuracy /
precision / recall / F1 from its four counts, then show on an imbalanced
set why accuracy alone lies. Pure Python, offline, deterministic.
"""

# --- The task -------------------------------------------------------------
# A support-ticket classifier decides "urgent" (1) vs "not urgent" (0).
# We have 10 tickets, each with a human GOLD label and the MODEL's PRED.
# gold[i] and pred[i] describe the same ticket.
GOLD = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
PRED = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]


def confusion(gold, pred):
    """Return (tp, fp, fn, tn) — the four cells of the 2x2, counted by hand."""
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:  # g == 0 and p == 0
            tn += 1
    return tp, fp, fn, tn


def accuracy(tp, fp, fn, tn):
    """Fraction of ALL predictions that were correct."""
    total = tp + fp + fn + tn
    return (tp + tn) / total if total else 0.0


def precision(tp, fp):
    """Of the ones we CALLED urgent, how many really were? Guard 0/0 -> 0.0."""
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp, fn):
    """Of the ones that WERE urgent, how many did we catch? Guard 0/0 -> 0.0."""
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(prec, rec):
    """Harmonic mean of precision and recall. Guard 0+0 -> 0.0."""
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def report(gold, pred, label):
    tp, fp, fn, tn = confusion(gold, pred)
    acc = accuracy(tp, fp, fn, tn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f = f1(prec, rec)
    print(f"{label}")
    print(f"  n = {len(gold)}  (positives in gold: {tp + fn}, negatives: {fp + tn})")
    print(f"  confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  accuracy  = (TP+TN)/n         = ({tp}+{tn})/{tp+fp+fn+tn} = {acc:.2f}")
    print(f"  precision = TP/(TP+FP)        = {tp}/({tp}+{fp}) = {prec:.2f}")
    print(f"  recall    = TP/(TP+FN)        = {tp}/({tp}+{fn}) = {rec:.2f}")
    print(f"  F1        = 2PR/(P+R)         = {f:.2f}")
    return acc, prec, rec, f


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 1 - WHAT IS A SCORE? metrics from the 2x2 confusion matrix.")
    print(line)

    report(GOLD, PRED, "A balanced set (5 urgent, 5 not):")

    print("\n" + "-" * 72)
    print("WHY ACCURACY LIES: an imbalanced set (2 urgent out of 20).")
    print("A lazy model that ALWAYS predicts 'not urgent' looks great on accuracy")
    print("but catches zero urgent tickets.")
    print("-" * 72)
    imb_gold = [1, 1] + [0] * 18            # 2 urgent, 18 not
    imb_pred = [0] * 20                      # always "not urgent"
    report(imb_gold, imb_pred, "Always-negative baseline:")
    print("\n  -> 90% accurate, 0% recall. Accuracy hid a model that is useless")
    print("     at the only job that matters. This is why we need the whole 2x2.")
    print(line)

# Expected output:
# ========================================================================
# PART 1 - WHAT IS A SCORE? metrics from the 2x2 confusion matrix.
# ========================================================================
# A balanced set (5 urgent, 5 not):
#   n = 10  (positives in gold: 5, negatives: 5)
#   confusion:  TP=3  FP=1  FN=2  TN=4
#   accuracy  = (TP+TN)/n         = (3+4)/10 = 0.70
#   precision = TP/(TP+FP)        = 3/(3+1) = 0.75
#   recall    = TP/(TP+FN)        = 3/(3+2) = 0.60
#   F1        = 2PR/(P+R)         = 0.67
#
# ------------------------------------------------------------------------
# WHY ACCURACY LIES: an imbalanced set (2 urgent out of 20).
# A lazy model that ALWAYS predicts 'not urgent' looks great on accuracy
# but catches zero urgent tickets.
# ------------------------------------------------------------------------
# Always-negative baseline:
#   n = 20  (positives in gold: 2, negatives: 18)
#   confusion:  TP=0  FP=0  FN=2  TN=18
#   accuracy  = (TP+TN)/n         = (0+18)/20 = 0.90
#   precision = TP/(TP+FP)        = 0/(0+0) = 0.00
#   recall    = TP/(TP+FN)        = 0/(0+2) = 0.00
#   F1        = 2PR/(P+R)         = 0.00
#
#   -> 90% accurate, 0% recall. Accuracy hid a model that is useless
#      at the only job that matters. This is why we need the whole 2x2.
# ========================================================================
