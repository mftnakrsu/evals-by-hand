# Part 10: Calibration

> A confidence is a promise: of the answers a model tags "80% sure," about 80% should be right. We bin 10 trivia predictions by confidence, build the reliability curve (mean confidence vs. accuracy per bin) by hand, then reduce it to two numbers, ECE and Brier score, run once for an over-confident model and once for a well-calibrated one on the exact same confidences.

[Read the essay](https://www.mefby.com/essays/calibration)

## What it covers
This part separates calibration from accuracy. Ten trivia questions each carry a stated confidence (0.6 to 1.0) and a correct/wrong outcome, scored twice against the same 10 confidences: once for an over-confident model whose outcomes do not track its confidence, and once for a well-calibrated model whose outcomes do. Predictions are grouped by hand into three confidence bins, the reliability curve (mean confidence vs. accuracy per bin) is built from those groups, and the whole picture is reduced to Expected Calibration Error (the weighted average gap between confidence and accuracy) and the Brier score (mean squared error of confidence against outcome). Same mean confidence (0.82) both times; ECE drops from 0.420 to 0.040 and Brier from 0.432 to 0.172 once the outcomes actually track the stated confidence.

## Files
- **`calibration.py`**: the single runnable script: bin assignment, the reliability curve, ECE, and Brier, run on both the over-confident and well-calibrated outcome sets, with the expected output included as a trailing comment block.
- **`calibration.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 calibration.py
```
Prefer it step by step? Open `calibration.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports, no API key, no network. Every number printed is deterministic and reproducible on any machine.
