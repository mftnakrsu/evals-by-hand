"""
Evals from First Principles - Part 15: A Tiny Eval Harness.

The capstone. The whole series assembles into one small, offline, pure-Python
harness with five reusable pieces:

  1. a labeled DATASET,
  2. a SCORER  -- a plain metric OR a (mock) LLM judge (Parts 1, 4),
  3. a BOOTSTRAP CI around the mean score (Part 6),
  4. a GATE: ship only if the CI lower bound clears a release bar (Part 11),
  5. a HARNESS that wires 1-4 together and prints a ship / hold verdict.

We then run the exact same harness on two tiny cases: a RAG-style QA eval
(a nod to RAG Part 8) and an agent-trajectory eval (a nod to Part 14 /
Agents Part 17). Swap the dataset or the scorer; everything downstream is
unchanged. Deterministic: the only randomness is the bootstrap, seeded.
"""

import numpy as np

# ==========================================================================
# PIECE 1 - the datasets (small enough to print every row)
# ==========================================================================

# A RAG-style QA set: a question, the GOLD short answer, the system's CANDIDATE.
# Item 6 is the interesting one: "Washington, D.C." is correct for the USA,
# but a strict string match will unfairly fail it. The judge will catch that.
QA = [
    {"q": "capital of France",    "gold": "Paris",     "cand": "Paris"},
    {"q": "capital of Japan",     "gold": "Tokyo",     "cand": "Tokyo"},
    {"q": "capital of Australia", "gold": "Canberra",  "cand": "Sydney"},
    {"q": "capital of Canada",    "gold": "Ottawa",    "cand": "Ottawa"},
    {"q": "capital of Brazil",    "gold": "Brasilia",  "cand": "Rio de Janeiro"},
    {"q": "capital of USA",       "gold": "Washington","cand": "Washington, D.C."},
    {"q": "capital of Turkey",    "gold": "Ankara",    "cand": "Ankara"},
    {"q": "capital of Spain",     "gold": "Madrid",    "cand": "Madrid"},
    {"q": "capital of India",     "gold": "New Delhi", "cand": "new delhi"},
    {"q": "capital of Italy",     "gold": "Rome",      "cand": "Milan"},
]

# An agent-trajectory set: for each task, the GOLD sequence of tool calls and
# the agent's ACTUAL sequence. We score how much of the path lined up (Part 14).
TRAJ = [
    {"task": "answer from a doc",   "gold": ["search", "read", "answer"],
                                     "pred": ["search", "read", "answer"]},
    {"task": "compute a total",     "gold": ["search", "calculator", "answer"],
                                     "pred": ["search", "weather", "answer"]},
    {"task": "summarize findings",  "gold": ["search", "read", "summarize", "answer"],
                                     "pred": ["search", "read", "answer"]},
    {"task": "quick lookup",        "gold": ["search", "answer"],
                                     "pred": ["search", "search", "answer"]},
    {"task": "schedule a meeting",  "gold": ["calendar", "email"],
                                     "pred": ["calendar", "email"]},
]


# ==========================================================================
# PIECE 2 - scorers: each maps one record to a score in [0, 1]
# ==========================================================================

def _norm(s):
    """Lowercase + strip: the cheapest possible answer normalization."""
    return s.strip().lower()


def exact_match(rec):
    """Strict string metric: 1.0 only if candidate equals gold after _norm."""
    return 1.0 if _norm(rec["cand"]) == _norm(rec["gold"]) else 0.0


def mock_judge(rec):
    """A deterministic stand-in for an LLM judge (Part 4). Its transparent
    rubric: 'correct if the gold answer is contained in the candidate, or vice
    versa' -- lenient enough to accept 'Washington, D.C.' for 'Washington'.
    A real judge would swap in behind a generate() call; the verdict stays 0/1."""
    g, c = _norm(rec["gold"]), _norm(rec["cand"])
    return 1.0 if (g in c or c in g) else 0.0


def trajectory_match(rec):
    """Step-level credit (Part 14): the fraction of positions where the gold
    and predicted tool calls agree, penalized by any length mismatch."""
    gold, pred = rec["gold"], rec["pred"]
    matches = sum(1 for i in range(min(len(gold), len(pred))) if gold[i] == pred[i])
    return matches / max(len(gold), len(pred))


# ==========================================================================
# PIECE 3 - the bootstrap CI (Part 6), the only randomness, seeded
# ==========================================================================

def bootstrap_ci(scores, rng, B=2000, alpha=0.05):
    """Resample the per-item scores WITH replacement B times, re-average each
    time, and read the 2.5 / 97.5 percentiles of those means. Returns (lo, hi)."""
    a = np.asarray(scores, dtype=float)
    n = len(a)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = a[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


# ==========================================================================
# PIECE 4 - the gate (Part 11): ship only if CI_lo clears the release bar
# ==========================================================================

def gate(ci_lo, bar):
    """The whole point of the series in one comparison: do NOT ship on a point
    estimate; ship only when the uncertainty band stays above the bar."""
    return "SHIP" if ci_lo >= bar else "HOLD"


# ==========================================================================
# PIECE 5 - the harness that wires 1-4 together
# ==========================================================================

def run_eval(name, dataset, scorer, bar, show, seed=0, B=2000):
    """Score every item, bootstrap a CI around the mean, and apply the gate.
    `show` prints one line per item so every number stays traceable."""
    scores = [scorer(rec) for rec in dataset]
    mean = sum(scores) / len(scores)
    rng = np.random.default_rng(seed)      # fixed seed -> reproducible CI
    lo, hi = bootstrap_ci(scores, rng, B)
    verdict = gate(lo, bar)

    print(f"  per-item score:")
    for rec, s in zip(dataset, scores):
        print("    " + show(rec, s))
    print(f"  n = {len(dataset)}   mean score = {mean:.2f}")
    print(f"  bootstrap 95% CI (B={B} resamples, seed={seed}): [{lo:.2f}, {hi:.2f}]")
    print(f"  gate: release bar = {bar:.2f} ; CI_lo = {lo:.2f} -> "
          f"{lo:.2f} {'>=' if lo >= bar else '<'} {bar:.2f} -> {verdict}")
    return {"name": name, "mean": mean, "lo": lo, "hi": hi,
            "bar": bar, "verdict": verdict}


def _show_qa(rec, s):
    return f"{rec['q']:<20} gold={rec['gold']:<11} cand={rec['cand']:<16} -> {int(s)}"


def _show_traj(rec, s):
    g = "[" + ",".join(rec["gold"]) + "]"
    p = "[" + ",".join(rec["pred"]) + "]"
    return f"{rec['task']:<19} gold={g:<34} pred={p:<27} -> {s:.2f}"


if __name__ == "__main__":
    line = "=" * 72
    dash = "-" * 72
    print(line)
    print("PART 15 - A TINY EVAL HARNESS: dataset + scorer + bootstrap CI + gate.")
    print(line)
    print("Five reusable pieces, one file:")
    print("  1. a labeled DATASET          2. a SCORER (a metric OR an LLM judge)")
    print("  3. a BOOTSTRAP CI (Part 6)    4. a GATE: ship iff CI_lo >= bar (Part 11)")
    print("  5. a HARNESS that wires 1-4 and prints a SHIP / HOLD verdict.")
    print("Swap the dataset or the scorer; everything downstream is unchanged.")

    results = []

    print("\n" + dash)
    print("RUN 1 - RAG-style QA eval (nod to RAG Part 8) | scorer = exact match")
    print(dash)
    results.append(run_eval("QA / exact-match", QA, exact_match, bar=0.35,
                            show=_show_qa))
    print("  read: 0.60 looks like a win, but on n=10 the CI floor (0.30) sits")
    print("        below the 0.35 bar, and strict matching failed 'Washington,")
    print("        D.C.' -- a correct answer. HOLD, and reach for a better scorer.")

    print("\n" + dash)
    print("RUN 2 - SAME QA dataset, SAME gate | scorer = mock LLM judge (Part 4)")
    print(dash)
    results.append(run_eval("QA / judge", QA, mock_judge, bar=0.35,
                            show=_show_qa))
    print("  read: the judge accepts 'Washington, D.C.' for 'Washington', so the")
    print("        mean rises to 0.70 and the CI floor to 0.40 -- now above 0.35.")
    print("        Same data, same bar: swapping the scorer flips HOLD into SHIP.")

    print("\n" + dash)
    print("RUN 3 - agent-trajectory eval (nod to Part 14 / Agents Part 17)")
    print(dash)
    results.append(run_eval("agent / trajectory", TRAJ, trajectory_match, bar=0.40,
                            show=_show_traj))
    print("  read: step-level credit averages to 0.70; the CI floor (0.47) clears")
    print("        the 0.40 bar. The identical harness gates a different domain.")

    print("\n" + line)
    print("SUMMARY - one harness, three evals")
    print(line)
    print(f"  {'eval':<20}{'mean':>6}{'95% CI':>16}{'bar':>7}{'verdict':>9}")
    for r in results:
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"  {r['name']:<20}{r['mean']:>6.2f}{ci:>16}{r['bar']:>7.2f}"
              f"{r['verdict']:>9}")
    print("  The gate reads the interval, never the point estimate. That single")
    print("  discipline -- CI_lo vs a bar -- is the whole series in one line.")
    print(line)

# Expected output:
# ========================================================================
# PART 15 - A TINY EVAL HARNESS: dataset + scorer + bootstrap CI + gate.
# ========================================================================
# Five reusable pieces, one file:
#   1. a labeled DATASET          2. a SCORER (a metric OR an LLM judge)
#   3. a BOOTSTRAP CI (Part 6)    4. a GATE: ship iff CI_lo >= bar (Part 11)
#   5. a HARNESS that wires 1-4 and prints a SHIP / HOLD verdict.
# Swap the dataset or the scorer; everything downstream is unchanged.
#
# ------------------------------------------------------------------------
# RUN 1 - RAG-style QA eval (nod to RAG Part 8) | scorer = exact match
# ------------------------------------------------------------------------
#   per-item score:
#     capital of France    gold=Paris       cand=Paris            -> 1
#     capital of Japan     gold=Tokyo       cand=Tokyo            -> 1
#     capital of Australia gold=Canberra    cand=Sydney           -> 0
#     capital of Canada    gold=Ottawa      cand=Ottawa           -> 1
#     capital of Brazil    gold=Brasilia    cand=Rio de Janeiro   -> 0
#     capital of USA       gold=Washington  cand=Washington, D.C. -> 0
#     capital of Turkey    gold=Ankara      cand=Ankara           -> 1
#     capital of Spain     gold=Madrid      cand=Madrid           -> 1
#     capital of India     gold=New Delhi   cand=new delhi        -> 1
#     capital of Italy     gold=Rome        cand=Milan            -> 0
#   n = 10   mean score = 0.60
#   bootstrap 95% CI (B=2000 resamples, seed=0): [0.30, 0.90]
#   gate: release bar = 0.35 ; CI_lo = 0.30 -> 0.30 < 0.35 -> HOLD
#   read: 0.60 looks like a win, but on n=10 the CI floor (0.30) sits
#         below the 0.35 bar, and strict matching failed 'Washington,
#         D.C.' -- a correct answer. HOLD, and reach for a better scorer.
#
# ------------------------------------------------------------------------
# RUN 2 - SAME QA dataset, SAME gate | scorer = mock LLM judge (Part 4)
# ------------------------------------------------------------------------
#   per-item score:
#     capital of France    gold=Paris       cand=Paris            -> 1
#     capital of Japan     gold=Tokyo       cand=Tokyo            -> 1
#     capital of Australia gold=Canberra    cand=Sydney           -> 0
#     capital of Canada    gold=Ottawa      cand=Ottawa           -> 1
#     capital of Brazil    gold=Brasilia    cand=Rio de Janeiro   -> 0
#     capital of USA       gold=Washington  cand=Washington, D.C. -> 1
#     capital of Turkey    gold=Ankara      cand=Ankara           -> 1
#     capital of Spain     gold=Madrid      cand=Madrid           -> 1
#     capital of India     gold=New Delhi   cand=new delhi        -> 1
#     capital of Italy     gold=Rome        cand=Milan            -> 0
#   n = 10   mean score = 0.70
#   bootstrap 95% CI (B=2000 resamples, seed=0): [0.40, 1.00]
#   gate: release bar = 0.35 ; CI_lo = 0.40 -> 0.40 >= 0.35 -> SHIP
#   read: the judge accepts 'Washington, D.C.' for 'Washington', so the
#         mean rises to 0.70 and the CI floor to 0.40 -- now above 0.35.
#         Same data, same bar: swapping the scorer flips HOLD into SHIP.
#
# ------------------------------------------------------------------------
# RUN 3 - agent-trajectory eval (nod to Part 14 / Agents Part 17)
# ------------------------------------------------------------------------
#   per-item score:
#     answer from a doc   gold=[search,read,answer]               pred=[search,read,answer]        -> 1.00
#     compute a total     gold=[search,calculator,answer]         pred=[search,weather,answer]     -> 0.67
#     summarize findings  gold=[search,read,summarize,answer]     pred=[search,read,answer]        -> 0.50
#     quick lookup        gold=[search,answer]                    pred=[search,search,answer]      -> 0.33
#     schedule a meeting  gold=[calendar,email]                   pred=[calendar,email]            -> 1.00
#   n = 5   mean score = 0.70
#   bootstrap 95% CI (B=2000 resamples, seed=0): [0.47, 0.93]
#   gate: release bar = 0.40 ; CI_lo = 0.47 -> 0.47 >= 0.40 -> SHIP
#   read: step-level credit averages to 0.70; the CI floor (0.47) clears
#         the 0.40 bar. The identical harness gates a different domain.
#
# ========================================================================
# SUMMARY - one harness, three evals
# ========================================================================
#   eval                  mean          95% CI    bar  verdict
#   QA / exact-match      0.60    [0.30, 0.90]   0.35     HOLD
#   QA / judge            0.70    [0.40, 1.00]   0.35     SHIP
#   agent / trajectory    0.70    [0.47, 0.93]   0.40     SHIP
#   The gate reads the interval, never the point estimate. That single
#   discipline -- CI_lo vs a bar -- is the whole series in one line.
# ========================================================================
