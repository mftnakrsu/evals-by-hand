# Part 2: Building a Golden Set

> A metric is only as good as the labeled set under it. This part builds a trustworthy eval set by hand on tiny, printable data: stratified vs. simple-random sampling on a rare class, the leakage trap where memorized near-duplicates inflate accuracy, and a rubric that turns "good answer?" into countable criteria.

[Read the essay](https://www.mefby.com/essays/building-a-golden-set)

## What it covers
This part builds the golden set every metric in the series depends on. A 40-ticket population with a rare "urgent" class (8 urgent, 32 not, true rate 0.20) shows why sampling method matters: a simple-random draw of 10 can land anywhere from 0 to 4 urgent tickets and misses the rare class entirely 7.4% of the time in a 10,000-trial simulation (cross-checked against the exact hypergeometric probability of 0.076), while stratified sampling pins the urgent share at exactly 2 every single draw, std 0. A second demonstration plants 4 near-duplicate "train" tickets inside a 10-item eval set: measured accuracy reads 0.800, but drops to the honest 0.667 once the memorized gimmes are removed. A third section turns "good answer?" into a 3-criteria, 0/1/2 rubric and scores three candidate answers against a pass-at-5 bar, with 2 of 3 clearing it.

## Files
- **`golden_set.py`**: the single runnable script: population and sampling (simple-random vs. stratified, with an exact hypergeometric cross-check), the leakage detector and honest-accuracy recomputation, and the rubric, with the expected output included as a trailing comment block.
- **`golden_set.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 golden_set.py
```
Prefer it step by step? Open `golden_set.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python and NumPy, no network, no API key. All randomness uses a fixed seed (`numpy.random.default_rng(0)`), so every number printed is deterministic and reproducible on any machine.
