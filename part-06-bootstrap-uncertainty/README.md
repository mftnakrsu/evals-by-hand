# Part 6: Uncertainty, Bootstrap Confidence Intervals by Hand

> A score is a point estimate, not the truth. We bootstrap a 30-item correct/incorrect set (21/30, accuracy 0.70) by resampling it with replacement 10000 times, read the 95% CI off the 2.5th and 97.5th percentiles, then show the same 0.70 accuracy tiled up to n=120 and n=480 to see the interval narrow as the set grows.

[Read the essay](https://www.mefby.com/essays/bootstrap-uncertainty)

## What it covers
This part turns a single reported score into an honest confidence interval. Starting from one eval run's 30 graded outputs (21 correct, accuracy 0.70), it resamples the set with replacement, with a fixed seed so every draw is traceable, and prints three individual resample index lists so the "with replacement" mechanic is visible before running the full B = 10000 bootstrap. The 95% CI at n=30 comes out to [0.53, 0.87], wide enough that a 2-point difference between two systems would be meaningless. Tiling the same set to n=120 and n=480 keeps the 0.70 accuracy fixed while the CI narrows by roughly half each time n quadruples, the ~1/sqrt(n) rule made concrete.

## Files
- **`bootstrap_ci.py`**: the single runnable script: the labeled set, the bootstrap resampling function, the percentile-based 95% CI, three inspectable example draws, the full B = 10000 bootstrap, and the n=30/120/480 scaling comparison, with the expected output included as a trailing comment block.
- **`bootstrap_ci.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 bootstrap_ci.py
```
Prefer it step by step? Open `bootstrap_ci.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python plus NumPy, no API key, no network. Every draw is seeded with `numpy.random.default_rng(0)`, so every number printed is deterministic and reproducible on any machine.
