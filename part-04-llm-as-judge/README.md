# Part 4: LLM-as-Judge by Hand

> Swap the human grader for a model and ask the same question: how often does the judge agree with human gold, and exactly where does it break? We build a rubric prompt, run a deterministic mock judge over 12 labeled items, parse its free text into PASS/FAIL/ABSTAIN, and score it against human gold with the same 2x2 confusion matrix from Part 1: accuracy 0.70, precision 0.80, recall 0.67, F1 0.73.

[Read the essay](https://www.mefby.com/essays/llm-as-judge)

## What it covers
This part replaces the human annotator from Part 3 with an LLM judge and measures it the same way: against trustworthy gold, not by assumption. A rubric prompt tells the judge to reply as `VERDICT: PASS` or `VERDICT: FAIL`, failing any refusal and passing only when the required key fact is present. A deterministic mock judge runs over 12 labeled items and returns raw free text, including the messy cases that break real judges: a tie when there is too little content to grade, and format drift when a rambling answer never emits a `VERDICT:` tag at all. A parser turns each raw verdict into PASS, FAIL, or ABSTAIN, dropping the two abstains and leaving 10 scorable items. The judge-vs-human confusion matrix on that scorable set gives TP=4, FP=1, FN=2, TN=3, and the one false positive (keyword-gaming) and two false negatives (correct answers the judge missed for using a synonym or a digit) show concretely why a judge needs to be measured, not trusted.

## Files
- **`llm_judge.py`**: the single runnable script: the rubric, the labeled set, the mock judge, the verdict parser, the failure-mode tally, and the confusion matrix and metrics, with the expected output included as a trailing comment block.
- **`llm_judge.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell.

## Run it
```bash
python3 llm_judge.py
```
Prefer it step by step? Open `llm_judge.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports, no API key, no network. The judge is a deterministic mock, not a real model call. Every number printed is deterministic and reproducible on any machine.
