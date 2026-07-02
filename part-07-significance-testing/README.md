# Part 7: Is the Difference Real? Paired Significance Testing by Hand

> Two models graded on the SAME 50-item eval set: Model A scores 82%, Model B scores 84%, a +2 point headline gap. We build the paired 2x2 of agreements and disagreements by hand, run McNemar's test on only the discordant items with an exact binomial p-value, then run a power/sample-size back-of-envelope showing how many items you would actually need to catch a 2-point gain 80% of the time.

[Read the essay](https://www.mefby.com/essays/significance-testing)

## What it covers
This part teaches why a headline score gap between two models is not automatically a real result. Because both models are graded on the same items, the comparison is paired: the two agreement cells (both right, both wrong) carry no signal at all, and everything worth testing lives in the discordant cells where the models split. McNemar's test treats that discordant count as a coin flip under the null of equal models, and we derive its exact two-sided p-value by hand, enumerating binomial coefficients, then sanity-check it against the chi-square approximation. The part closes with a power calculation that shows a 50-item eval has almost no chance of detecting a true 2-point gain, and that pairing meaningfully lowers the sample size you would need compared to treating the two models as independent.

## Files
- **`significance.py`**: the single runnable script: the paired 2x2, McNemar's test (chi-square and exact binomial), and the unpaired vs. paired sample-size back-of-envelope, with the expected output included as a trailing comment block.
- **`significance.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 significance.py
```
Prefer it step by step? Open `significance.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python + NumPy, no API key, no network. A fixed seed (`numpy.random.default_rng(0)`) makes every number printed deterministic and reproducible on any machine.
