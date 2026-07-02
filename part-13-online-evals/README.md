# Part 13: Online Evals. An A/B Test With a Guardrail, by Hand

> Offline scores are necessary, not sufficient. The only place a change proves itself is production, under a controlled A/B test. Here we run one by hand: a support chatbot's new prompt against the old one, a primary business metric (resolution rate) tested for significance with a two-proportion z-test derived from scratch, and a guardrail metric (refusal rate) that must not regress even when the primary metric wins.

[Read the essay](https://www.mefby.com/essays/online-evals)

## What it covers
This part moves the eval out of the lab and into live traffic. A support chatbot ships a new answer-generation prompt and splits one week of sessions 50/50 between control (old prompt) and treatment (new prompt): 400 sessions each, with resolved and refused counts for both arms. The primary metric, resolution rate, shows a large observed lift (168/400 to 200/400, +8.0 pp), and a two-proportion z-test built by hand from the pooled rate, standard error, and z-score turns that lift into a p-value (0.0232) that clears alpha=0.05. But a guardrail metric, refusal rate, doubles from 20/400 to 40/400 (+5.0 pp), breaching a pre-agreed +2 pp margin. The ship decision needs both conditions at once, and here it lands on DO NOT SHIP: a real, significant primary gain that is not enough on its own once the guardrail has failed.

## Files
- **`online_evals.py`**: the single runnable script: the scenario data, the `rate`, `normal_cdf`, `two_proportion_ztest`, and `bar` helpers, and the full primary metric / significance test / guardrail / verdict walkthrough, with the expected output included as a trailing comment block.
- **`online_evals.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell, reproducing the same numbers.

## Run it
```bash
python3 online_evals.py
```
Prefer it step by step? Open `online_evals.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python plus the standard library (`math.erf` gives the exact standard-normal CDF, no SciPy needed). No API key, no network. Every number printed is deterministic and reproducible on any machine.
