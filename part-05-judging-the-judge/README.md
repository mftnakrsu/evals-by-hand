# Part 5: Judging the Judge

> An LLM judge is itself a model, so it has systematic, measurable biases. We build a deterministic mock judge with two baked-in flaws, a position bias and a verbosity bias, measure each on a dozen paired comparisons, then correct the position bias by running both orderings and averaging.

[Read the essay](https://www.mefby.com/essays/judging-the-judge)

## What it covers
This part treats the LLM-as-judge from Part 4 as a model in its own right, one with systematic errors you can measure the same way you measure any model's errors: with controlled probes and counting. A deterministic mock judge scores each answer as its honest quality, plus a fixed bonus for sitting in slot A (position bias), plus a per-word bonus (verbosity bias). On 12 paired A-vs-B comparisons between system X and system Y, swapping who goes first flips the verdict on 7 of 12 pairs (0.58). A second probe of 5 pairs with tied quality shows the longer answer wins every time (1.00). Running both orderings and averaging cancels the position bonus exactly, because every answer sits in slot A once, cutting disagreement with the true winner from 5/12 (0.42) to 1/12 (0.08) and dropping the corrected judge's position-bias rate to exactly 0/12. The one survivor is a verbosity error: averaging fixes where an answer sits, not how long it is.

## Files
- **`judge_bias.py`**: the single runnable script: the biased mock judge, the position-bias sweep, the verbosity probe, and the order-averaging correction, with the expected output included as a trailing comment block.
- **`judge_bias.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 judge_bias.py
```
Prefer it step by step? Open `judge_bias.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports beyond `collections.namedtuple`, no API key, no network. Every number printed is deterministic and reproducible on any machine.
