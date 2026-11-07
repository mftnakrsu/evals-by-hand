# Part 9: Arenas. Elo and Bradley-Terry by Hand

> A leaderboard turns thousands of pairwise "which answer is better?" battles into one ranking. We build that turn by hand, two ways, on the same four chatbots and the same 18 battles: Elo, an online update replayed after each battle, and Bradley-Terry, a maximum-likelihood fit over the aggregate win matrix. Replaying the identical battles in two different orders makes Elo disagree with itself, while Bradley-Terry lands on the same ranking every time.

[Read the essay](https://www.mefby.com/essays/arenas-elo-bradley-terry)

## What it covers
This part builds a round-robin arena for four chatbots (A, B, C, D), each pair battling 3 times for 18 battles total, judged winner by winner into a win matrix with a couple of upsets baked in (B beats A once, C beats B once, D beats C once). That win matrix is turned into a leaderboard two ways. First Elo: one hand update (A beats B, both starting at 1000, K=32) shows a coin-flip expected score moving the rating by 16 points, then all 18 battles are replayed in the order listed and again in a fixed-seed shuffled order, and the final ratings disagree: Elo is order-dependent. Then Bradley-Terry: the Zermelo maximum-likelihood iteration fits strengths to the aggregate win matrix alone, converging to a ranking that cannot depend on order because it never sees one, plus calibrated head-to-head win probabilities read straight off the fitted strengths. Finally a bootstrap asks how far to trust that ranking: each pair's 3 battles are redrawn with replacement 10,000 times (seed 0), Bradley-Terry is refit with the same 20 MM iterations each time, and the script prints a 95% interval for every strength and how often each part of the order survives. Every neighboring pair of intervals overlaps, and the full order `A > B > C > D` holds in only about half of the refits.

## Files
- **`arenas.py`**: the single runnable script: the Elo update rule, replaying all 18 battles in two orders, the Bradley-Terry MM iteration on the aggregate win matrix, and the bootstrap intervals for that fit, with the expected output included as a trailing comment block.
- **`arenas.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 arenas.py
```
Prefer it step by step? Open `arenas.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports beyond `random` for a fixed-seed shuffle and the bootstrap resamples, no API key, no network. Every number printed is deterministic and reproducible on any machine.

One note on the bootstrap: the figure in the essay drew its resamples with numpy's `default_rng(0)` (PCG64), while this script uses the standard library's `random.Random(0)`, a different random stream. The intervals come out the same to 4 decimals; the survival counts differ by a few dozen per 10,000 (the full order holds 5410 times here, 5426 in the figure).
