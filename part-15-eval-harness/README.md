# Part 15: A Tiny Eval Harness

> The capstone: the whole series assembled into one small, offline, pure-Python harness with five reusable pieces (dataset, scorer, bootstrap CI, gate, harness), run three ways on two tiny datasets.

[Read the essay](https://www.mefby.com/essays/eval-harness)

## What it covers
This part wires every earlier piece of the series into one file. A 10-item RAG-style capital-cities QA set (a nod to RAG Part 8) is scored first with strict exact match, then with a lenient mock LLM judge (Part 4); a 5-task agent tool-call trajectory set (a nod to Part 14 / Agents Part 17) is scored with step-level trajectory match. Each run bootstraps a 95% confidence interval (Part 6, seed 0, B=2000) around the mean and applies a gate (Part 11) that ships only if the CI lower bound clears a release bar. Swapping just the scorer, exact match to judge, on the identical QA data flips the verdict from HOLD to SHIP, and the same harness gates a completely different domain, agent trajectories, without changing a line.

## Files
- **`eval_harness.py`**: the single runnable script: the datasets, the scorers, the bootstrap CI, the gate, the harness, and all three runs plus the summary table, with the expected output included as a trailing comment block.
- **`eval_harness.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 eval_harness.py
```
Prefer it step by step? Open `eval_harness.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python + NumPy, no other imports, no API key, no network. Every number printed is deterministic and reproducible on any machine (the bootstrap uses a fixed seed).
