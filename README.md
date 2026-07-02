# evals-by-hand

Build an LLM **evaluation** stack from first principles: one runnable Python
file per concept, no frameworks hiding the moving parts. Companion code for the
**Evals from First Principles** series on
[mefby.com](https://www.mefby.com/essays/evals): the third by-hand series,
after [rag-by-hand](https://github.com/mftnakrsu/rag-by-hand) and
[agents-by-hand](https://github.com/mftnakrsu/agents-by-hand). The series
builds the measurement stack that tells a real 2% gain from noise: metrics and
the confusion matrix, golden sets, annotator agreement, LLM-as-judge and its
biases, confidence intervals, significance and power, pass@k, arenas,
calibration, and a CI regression gate.

> "Build it by hand, understand every line."

There, retrieval was the whole system; here, the **agent** was the system and
retrieval was one tool it called. In this series, the **evaluation** is the
system: every metric, test, and gate is derived from scratch on a small
labeled set you can trace end to end, not imported from a library and taken on
faith. Each folder maps 1:1 to an essay.

Every part ships **two ways to learn the same concept**: a single runnable
`.py` (the whole idea, top to bottom, a self-contained single-file read) and a
step-by-step **Jupyter notebook** (`.ipynb`) that rebuilds it cell by cell,
with the *why* spelled out before each small step. Both run **offline, with no
API key and no framework**: pure Python plus NumPy is the reproducible
default, and the LLM-as-judge parts fall back to a deterministic mock judge,
with a real provider one env flag away via a single swappable `generate()`.

## The series

| Part | Topic | Code | Notebook | Essay |
|---|---|---|---|---|
| 1 | What Is a Score? metrics and the confusion matrix by hand | [scoring_basics.py](part-01-what-is-a-score/scoring_basics.py) | [notebook](part-01-what-is-a-score/scoring_basics.ipynb) | [read](https://www.mefby.com/essays/what-is-a-score) |
| 2 | Building a Golden Set: sampling, stratification, leakage, rubrics | [golden_set.py](part-02-building-a-golden-set/golden_set.py) | [notebook](part-02-building-a-golden-set/golden_set.ipynb) | [read](https://www.mefby.com/essays/building-a-golden-set) |
| 3 | Do Humans Even Agree? Cohen's and Fleiss' kappa by hand | [agreement.py](part-03-annotator-agreement/agreement.py) | [notebook](part-03-annotator-agreement/agreement.ipynb) | [read](https://www.mefby.com/essays/annotator-agreement) |
| 4 | LLM-as-Judge by Hand: rubric, parsing, the judge confusion matrix | [llm_judge.py](part-04-llm-as-judge/llm_judge.py) | [notebook](part-04-llm-as-judge/llm_judge.ipynb) | [read](https://www.mefby.com/essays/llm-as-judge) |
| 5 | Judging the Judge: position, verbosity, self-preference bias | [judge_bias.py](part-05-judging-the-judge/judge_bias.py) | [notebook](part-05-judging-the-judge/judge_bias.ipynb) | [read](https://www.mefby.com/essays/judging-the-judge) |
| 6 | Uncertainty: bootstrap confidence intervals by hand | [bootstrap_ci.py](part-06-bootstrap-uncertainty/bootstrap_ci.py) | [notebook](part-06-bootstrap-uncertainty/bootstrap_ci.ipynb) | [read](https://www.mefby.com/essays/bootstrap-uncertainty) |
| 7 | Is the Difference Real? McNemar, paired tests, and power | [significance.py](part-07-significance-testing/significance.py) | [notebook](part-07-significance-testing/significance.ipynb) | [read](https://www.mefby.com/essays/significance-testing) |
| 8 | pass@k: deriving the unbiased estimator | [pass_at_k.py](part-08-pass-at-k/pass_at_k.py) | [notebook](part-08-pass-at-k/pass_at_k.ipynb) | [read](https://www.mefby.com/essays/pass-at-k) |
| 9 | Arenas: Elo and Bradley-Terry by hand | [arenas.py](part-09-arenas-elo-bradley-terry/arenas.py) | [notebook](part-09-arenas-elo-bradley-terry/arenas.ipynb) | [read](https://www.mefby.com/essays/arenas-elo-bradley-terry) |
| 10 | Calibration: reliability curves, ECE, and Brier score | [calibration.py](part-10-calibration/calibration.py) | [notebook](part-10-calibration/calibration.ipynb) | [read](https://www.mefby.com/essays/calibration) |
| 11 | Regression Gates: CI gates, contamination, and drift | [regression_gates.py](part-11-regression-gates/regression_gates.py) | [notebook](part-11-regression-gates/regression_gates.ipynb) | [read](https://www.mefby.com/essays/regression-gates) |
| 12 | The Eval Flywheel: mining production logs into eval sets | [flywheel.py](part-12-eval-flywheel/flywheel.py) | [notebook](part-12-eval-flywheel/flywheel.ipynb) | [read](https://www.mefby.com/essays/eval-flywheel) |
| 13 | Online Evals: A/B tests and guardrail metrics | [online_evals.py](part-13-online-evals/online_evals.py) | [notebook](part-13-online-evals/online_evals.ipynb) | [read](https://www.mefby.com/essays/online-evals) |
| 14 | Agent-Trajectory Evals: grading the path, not just the answer | [trajectory_evals.py](part-14-agent-trajectory-evals/trajectory_evals.py) | [notebook](part-14-agent-trajectory-evals/trajectory_evals.ipynb) | [read](https://www.mefby.com/essays/agent-trajectory-evals) |
| 15 | A Tiny Eval Harness: dataset, metric, judge, CI, gate | [eval_harness.py](part-15-eval-harness/eval_harness.py) | [notebook](part-15-eval-harness/eval_harness.ipynb) | [read](https://www.mefby.com/essays/eval-harness) |

## Quick start

```bash
# from the repo root; offline, no API key, no network (Part 1 needs only the standard library)
python3 part-01-what-is-a-score/scoring_basics.py
```

Prefer it step by step? Open the part's `.ipynb` in Jupyter or Colab.

Optional real paths (any one):
```bash
export OPENAI_API_KEY=...        # see the real LLM-as-judge path
# the deterministic mock judge still runs so output stays reproducible
```

## How the offline fallback works

The teaching point of every part is something you can *read*: the scoring
logic is a transparent, hand-derived computation in the offline default, so
you can see exactly why a score comes out the way it does. With an API key,
the judge parts route through a real model via `generate()`, but the file
always falls through to the deterministic mock judge so the printed trace
stays reproducible. SDK names and model ids move fast; only `generate()` ever
needs editing to light up the real path.

## License

MIT, see [LICENSE](LICENSE).
