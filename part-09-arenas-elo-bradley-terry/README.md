# Part 9: Arenas. Elo and Bradley-Terry by Hand

> A leaderboard turns thousands of pairwise "which answer is better?" battles into one ranking. We build that turn by hand, two ways, on the same four chatbots and the same 18 battles: Elo, an online update replayed after each battle, and Bradley-Terry, a maximum-likelihood fit over the aggregate win matrix. Replaying the identical battles in two different orders makes Elo disagree with itself, while Bradley-Terry lands on the same ranking every time.

[Read the essay](https://www.mefby.com/essays/arenas-elo-bradley-terry)

## What it covers
This part builds a round-robin arena for four chatbots (A, B, C, D), each pair battling 3 times for 18 battles total, judged winner by winner into a win matrix with a couple of upsets baked in (B beats A once, C beats B once, D beats C once). That win matrix is turned into a leaderboard two ways. First Elo: one hand update (A beats B, both starting at 1000, K=32) shows a coin-flip expected score moving the rating by 16 points, then all 18 battles are replayed in the order listed and again in a fixed-seed shuffled order, and the final ratings disagree: Elo is order-dependent. Then Bradley-Terry: the Zermelo maximum-likelihood iteration fits strengths to the aggregate win matrix alone, converging to a ranking that cannot depend on order because it never sees one, plus calibrated head-to-head win probabilities read straight off the fitted strengths.

## Files
- **`arenas.py`**: the single runnable script: the Elo update rule, replaying all 18 battles in two orders, and the Bradley-Terry MM iteration on the aggregate win matrix, with the expected output included as a trailing comment block.
- **`arenas.ipynb`**: step-by-step notebook: a markdown why before each code step, built cell by cell.

## Run it
```bash
python3 arenas.py
```
Prefer it step by step? Open `arenas.ipynb` in Jupyter or Colab.

## Offline by design
Pure Python standard library, no imports beyond `random` for a fixed-seed shuffle, no API key, no network. Every number printed is deterministic and reproducible on any machine.
