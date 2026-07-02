# Part 14: Agent-Trajectory Evals. Grade the Path, Not Just the Answer

> Agents Part 17 graded whether the agent's final answer was right. Here we grade the path: the sequence of tool calls the agent took to get there. On a reference trajectory (search, open, convert, answer) versus eight actual agent runs, we compute three trajectory-match scores, assign step-level credit, then grade a pass/fail rubric with a mock judge and measure judge-vs-human agreement with Cohen's kappa.

[Read the essay](https://www.mefby.com/essays/agent-trajectory-evals)

## What it covers
An agent must answer "what does 3 nights at the top-rated Paris hotel cost in USD?", whose approved reference trajectory is the tool sequence search, open, convert, answer. Eight actual agent runs (R1 to R8, for example R2 has the right tools in the wrong order, R3 skips the currency conversion and re-searches instead, R5 is lazy, R8 just guesses) are each scored four ways: exact-match, order-aware via longest common subsequence, set-overlap via Jaccard, and step-level credit from an LCS alignment that labels every step MATCH, EXTRA, or MISSING. A pass/fail rubric is then graded by a deterministic mock judge and a human, and their agreement is chance-corrected with Cohen's kappa (0.75 raw agreement, but only 0.50 kappa once chance is removed). The two disagreements are exactly the wrong-order and skipped-step runs the lenient judge waved through: grading the trajectory catches what grading only the final answer would miss.

## Files
- **`trajectory_evals.py`**: the single runnable script: the reference trajectory and the eight runs, the `lcs_pairs`, `exact_match`, `order_aware`, `set_overlap`, `step_credit`, and `labeled_steps` scoring functions, the mock `judge` and `cohen_kappa`, and the full four-part walkthrough, with the expected output included as a trailing comment block.
- **`trajectory_evals.ipynb`**: step-by-step notebook: a markdown why before each small code step, built cell by cell, reproducing the same numbers.

## Run it
```bash
python3 trajectory_evals.py
```
Prefer it step by step? Open `trajectory_evals.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports, no API key, no network. The mock judge is a transparent rule standing in for an LLM grader. Every number printed is deterministic and reproducible on any machine.
