# Part 12: The Eval Flywheel

> Your best eval set is not written once, it is mined from production. We take 12 mock support-assistant traces, rank them by a cheap priority score built only from free signals (confidence, guardrail flag), spend a fixed annotation budget two ways, and watch a golden set grow exactly where the model is weakest, then close the loop honestly about what that golden set does and does not tell you.

[Read the essay](https://www.mefby.com/essays/eval-flywheel)

## What it covers
This part turns a production log into a golden set. Each of 12 traces carries two cheap, always-visible signals (the model's self-reported confidence and a guardrail or thumbs-down flag) while its true correct or failure label stays hidden until a human pays to annotate it. A priority score built only from the cheap signals ranks the traces by suspicion with zero labels required. Spending a budget of 6 annotations on the top-priority traces (hard-case sampling) catches all 4 real failures, while a fixed-seed random draw of the same size catches only 2, roughly 2x the failure density for the same cost. Running the flywheel in three rounds of 2 annotations shows the golden set's failure coverage climbing to 1.00 while its own failure rate settles well above the true production rate, a reminder that an enriched golden set is a diagnostic tool, not a random sample of production, and needs a small random holdout alongside it to estimate the real rate without bias.

## Files
- **`flywheel.py`**: the single runnable script: the production log, the priority score, hard-case vs. random annotation, the three-round flywheel, and the honest close, with the expected output included as a trailing comment block.
- **`flywheel.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 flywheel.py
```
Prefer it step by step? Open `flywheel.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no third-party imports, no API key, no network. The only randomness is a fixed-seed draw (`random.Random(0)`), so every number printed is deterministic and reproducible on any machine.
