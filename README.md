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
