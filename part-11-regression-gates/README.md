# Part 11: Regression Gates

> Everything the series built (a score, a golden set, a judge, a confidence interval) becomes one decision a CI pipeline can make on its own: ship, or block. We run three cheap checks on two candidate runs of the same 20-item eval set: a bootstrap CI threshold gate, an n-gram contamination scan, and a Population Stability Index drift check, then let the numbers decide.

[Read the essay](https://www.mefby.com/essays/regression-gates)

## What it covers
This part turns the series' tools into a gate you would actually stake a deploy on. Two candidate runs of the same 20-item eval set are compared: run A (mean 0.900, a clear pass) and run B (mean 0.725, an ambiguous one that games the point estimate). Check 1 reuses the Part 6 bootstrap to build a 95% confidence interval and passes a run only if the interval's lower bound clears a 0.70 bar, not just its mean. Check 2 scans four eval questions against a four-document training corpus with word-trigram overlap to catch a leaked eval item (one question was pasted directly into training). Check 3 buckets both runs' score distributions and compares them with the Population Stability Index, catching a distribution shift that a mean alone would hide. Run B's mean clears the bar but its CI does not, and its distribution drifts significantly from run A, so the gate blocks the deploy.

## Files
- **`regression_gates.py`**: the single runnable script: the bootstrap CI threshold gate, the n-gram contamination scan, the PSI drift check, and the combined gate verdict, with the expected output included as a trailing comment block.
- **`regression_gates.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 regression_gates.py
```
Prefer it step by step? Open `regression_gates.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python + NumPy, no other imports, no API key, no network. Every number printed is deterministic and reproducible on any machine (the bootstrap uses a fixed seed).
