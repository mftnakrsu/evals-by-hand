# Part 8: pass@k, an Unbiased Estimator by Hand

> The code-gen and agent metric everyone quotes and few compute correctly. We sample n=5 completions for one task, c=2 pass the hidden unit tests, and we want pass@k for a budget of k=2 draws. The tempting plug-in 1-(1-c/n)^k is biased; the unbiased combinatorial estimator 1-C(n-c,k)/C(n,k) is checked against a brute-force enumeration of every size-k subset, then proven unbiased exactly by taking expectations over c ~ Binomial(n, p) in rational arithmetic.

[Read the essay](https://www.mefby.com/essays/pass-at-k)

## What it covers
This part derives pass@k, the standard metric for "did at least one of k sampled completions pass," from first principles on a single worked task: 5 sampled completions, 2 of which pass a hidden unit-test suite, with a budget of k=2 draws. It shows why the obvious plug-in estimator (treat each sample as an independent coin with rate c/n) is biased, because it silently samples with replacement from a pool you actually drew once, without replacement. It derives the unbiased estimator as a counting argument over C(n,k) equal-probability subsets, checks it against a brute-force enumeration of all 10 possible size-2 draws, sweeps every budget k=1..5, and finally proves the plug-in's bias exactly (no simulation) by taking the exact expectation of both estimators over c ~ Binomial(n, p) in exact rational arithmetic.

## Files
- **`pass_at_k.py`**: the single runnable script: both estimators, the brute-force check, the k=1..5 sweep, and the exact bias proof, with the expected output included as a trailing comment block.
- **`pass_at_k.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 pass_at_k.py
```
Prefer it step by step? Open `pass_at_k.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library (math, itertools, fractions), no imports beyond that, no API key, no network. No sampling either: every number is exact and reproducible on any machine.
