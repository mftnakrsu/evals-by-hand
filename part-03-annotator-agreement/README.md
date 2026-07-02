# Part 3: Do Humans Even Agree? Annotator Agreement by Hand

> Your labels have noise; measure it before you trust them. We build the 2x2 by hand from two raters on 20 support tickets, collapse a scary-high 90% raw agreement into Cohen's kappa (0.61) once chance is corrected for, then extend to three raters with Fleiss' kappa (0.44), and close with a one-line note on Krippendorff's alpha as the general case.

[Read the essay](https://www.mefby.com/essays/annotator-agreement)

## What it covers
This part questions the gold labels the whole series has been treating as ground truth. Two human annotators grade the same 20 support tickets as urgent or not urgent, and agree on 18 of 20, a scary-high 90% raw agreement. The script fills the 2x2 by hand, computes each rater's marginal rate, derives chance agreement `pe = 0.7450`, and shows that the honest, chance-corrected number is Cohen's kappa `0.61`, far lower than the raw percentage suggested. It then extends the same idea to three raters on five fresh tickets with Fleiss' kappa (`0.44`), and closes with a note on Krippendorff's alpha as the general case for any number of raters, missing labels, and any scale.

## Files
- **`agreement.py`**: the single runnable script: `cohen_cells`, `cohen_kappa`, and `fleiss_kappa` built from scratch, with both worked examples and the expected output included as a trailing comment block.
- **`agreement.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 agreement.py
```
Prefer it step by step? Open `agreement.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports, no API key, no network. Every number printed is deterministic and reproducible on any machine.
