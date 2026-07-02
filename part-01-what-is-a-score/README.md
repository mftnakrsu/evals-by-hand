# Part 1: What Is a Score? Metrics and the Confusion Matrix by Hand

> The atoms of every eval: a task, a gold label, and a metric. We build the 2x2 confusion matrix by hand from 10 graded outputs, derive accuracy, precision, recall, and F1 from its four counts, then show on an imbalanced set why accuracy alone lies.

[Read the essay](https://www.mefby.com/essays/what-is-a-score)

## What it covers
This part builds the measurement primitive every later part in the series depends on: a support-ticket classifier's 10 graded outputs (gold label vs. model prediction) counted by hand into a 2x2 confusion matrix (TP, FP, FN, TN), then accuracy, precision, recall, and F1 derived from just those four numbers. A second, imbalanced set (2 urgent tickets out of 20) scores an always-negative baseline at 90% accuracy and 0% recall, showing concretely why a single accuracy number can hide a model that is useless at the one job it exists to do.

## Files
- **`scoring_basics.py`**: the single runnable script: the confusion matrix, the four metrics, and both worked examples (balanced set, then the class-imbalance demonstration), with the expected output included as a trailing comment block.
- **`scoring_basics.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 scoring_basics.py
```
Prefer it step by step? Open `scoring_basics.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports, no API key, no network. Every number printed is deterministic and reproducible on any machine.
