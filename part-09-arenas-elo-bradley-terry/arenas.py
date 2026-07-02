"""
Evals from First Principles — Part 9: Arenas (Elo & Bradley-Terry)

A leaderboard turns thousands of pairwise "which answer is better?" battles into
one ranking. We do it two ways by hand on the SAME four chatbots (A, B, C, D):

  1. Elo — an ONLINE update after each battle. Expected score
     E = 1/(1+10^((Rb-Ra)/400)), then R' = R + K*(S - E). We replay the exact
     same battles in two different orders and watch the final ratings disagree:
     Elo is ORDER-DEPENDENT.

  2. Bradley-Terry — fit ALL battles at once by maximum likelihood. It sees only
     the aggregate win matrix, so it has no notion of order: the ranking it
     converges to is ORDER-INDEPENDENT.

Pure Python, offline, deterministic. Any shuffle uses a fixed seed.
"""

import random

MODELS = ["A", "B", "C", "D"]

# --- The battle record ----------------------------------------------------
# Four chatbots fought a round-robin: every pair battled 3 times. WINS[i][j] is
# the number of times model i beat model j (a "win" = the judge preferred i).
# Read across a row to see who i beat; down a column to see who beat i.
#         beat->   A  B  C  D
WINS = [
    [0, 2, 3, 3],   # A beat B twice, C 3x, D 3x   (8 wins, 1 loss)
    [1, 0, 2, 3],   # B beat A once, C 2x, D 3x    (6 wins, 3 losses)
    [0, 1, 0, 2],   # C beat B once, D 2x          (3 wins, 6 losses)
    [0, 0, 1, 0],   # D beat C once                (1 win,  8 losses)
]
# True strength order baked in: A > B > C > D, with a couple of upsets.


# --- 1. Elo ---------------------------------------------------------------
def expected(ra, rb):
    """Probability the A-side scores, given ratings ra and rb (logistic, /400)."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(ra, rb, score_a, k=32):
    """One battle. score_a is 1 if A won, 0 if A lost. Return (ra', rb')."""
    ea = expected(ra, rb)
    ra_new = ra + k * (score_a - ea)
    rb_new = rb + k * ((1 - score_a) - (1 - ea))
    return ra_new, rb_new


def battle_log():
    """Flatten the win matrix into a list of (winner, loser) battles."""
    games = []
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            for _ in range(WINS[i][j]):
                games.append((i, j))
    return games


def run_elo(games, k=32, start=1000.0):
    """Replay a sequence of (winner, loser) battles; return final ratings."""
    r = [start] * len(MODELS)
    for w, l in games:
        rw, rl = elo_update(r[w], r[l], 1.0, k)
        r[w], r[l] = rw, rl
    return r


# --- 2. Bradley-Terry (MLE by the MM / Zermelo iteration) -----------------
def bt_fit(wins, iters=20):
    """
    Fit strengths p so that P(i beats j) = p_i / (p_i + p_j) is most likely.
    Update rule (Zermelo 1929): p_i <- W_i / sum_j n_ij/(p_i+p_j), where W_i is
    i's total wins and n_ij the games between i and j. Normalize sum(p)=1 each
    step. Depends ONLY on the aggregate matrix -> order cannot matter.
    Returns (final_p, snapshots) where snapshots are p after selected iters.
    """
    n = len(wins)
    total_wins = [sum(wins[i]) for i in range(n)]
    games_between = [[wins[i][j] + wins[j][i] for j in range(n)] for i in range(n)]
    p = [1.0 / n] * n
    snapshots = {0: p[:]}
    for step in range(1, iters + 1):
        new = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j != i:
                    denom += games_between[i][j] / (p[i] + p[j])
            new[i] = total_wins[i] / denom
        s = sum(new)
        p = [x / s for x in new]
        if step in (1, 2, 3, 20):
            snapshots[step] = p[:]
    return p, snapshots


def ranking(scores):
    """Return model letters ordered best-first by score."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return " > ".join(MODELS[i] for i in order)


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 9 - ARENAS: from pairwise battles to one ranking.")
    print(line)

    # -- One Elo update, fully by hand --------------------------------------
    print("ONE ELO UPDATE (A beats B, both start at 1000, K=32):")
    ea = expected(1000.0, 1000.0)
    print(f"  E_A = 1/(1+10^((Rb-Ra)/400)) = 1/(1+10^0) = {ea:.2f}")
    print(f"  A won, so S_A = 1.  R_A' = 1000 + 32*(1 - {ea:.2f}) "
          f"= {1000 + 32 * (1 - ea):.1f}")
    print(f"           S_B = 0.  R_B' = 1000 + 32*(0 - {ea:.2f}) "
          f"= {1000 + 32 * (0 - (1 - ea)):.1f}")
    print("  Equal ratings -> a coin-flip prediction -> the win moves 16 points.")

    # -- Elo is order-dependent --------------------------------------------
    games = battle_log()
    print("\n" + "-" * 72)
    print(f"ELO OVER ALL {len(games)} BATTLES (start 1000, K=32).")
    print("Same battles, two orders:")
    print("-" * 72)

    r_asplayed = run_elo(games)
    shuffled = games[:]
    random.Random(0).shuffle(shuffled)
    r_shuffled = run_elo(shuffled)

    print("  as-listed order:  " +
          "  ".join(f"{m}={r:.1f}" for m, r in zip(MODELS, r_asplayed)))
    print("  shuffled order:   " +
          "  ".join(f"{m}={r:.1f}" for m, r in zip(MODELS, r_shuffled)))
    diffs = [b - a for a, b in zip(r_asplayed, r_shuffled)]
    print("  difference:       " +
          "  ".join(f"{m}={d:+.1f}" for m, d in zip(MODELS, diffs)))
    print(f"  ranking (as-listed): {ranking(r_asplayed)}")
    print(f"  ranking (shuffled):  {ranking(r_shuffled)}")
    print("  -> Same battles, different NUMBERS: Elo depends on match order.")

    # -- Bradley-Terry fits everything at once -----------------------------
    print("\n" + "-" * 72)
    print("BRADLEY-TERRY MLE on the aggregate win matrix (iterated):")
    print("-" * 72)
    print("  win matrix (row beat col):   A  B  C  D")
    for i, m in enumerate(MODELS):
        print(f"                          {m}  [ " +
              "  ".join(str(WINS[i][j]) for j in range(len(MODELS))) + " ]")

    p, snaps = bt_fit(WINS)
    print("\n  strengths p (normalized to sum 1), by iteration:")
    print("                    A       B       C       D")
    for step in (0, 1, 2, 3, 20):
        row = snaps[step]
        tag = "start" if step == 0 else f"it {step}"
        print(f"    {tag:>6}      " + "  ".join(f"{x:.4f}" for x in row))
    print(f"\n  converged ranking: {ranking(p)}")
    print(f"  P(A beats D) = pA/(pA+pD) = {p[0] / (p[0] + p[3]):.3f}")
    print(f"  P(B beats C) = pB/(pB+pC) = {p[1] / (p[1] + p[2]):.3f}")
    print("  -> No order anywhere: BT reads only the totals, so its ranking is")
    print("     the same no matter how the battles were shuffled.")
    print(line)

# Expected output:
# ========================================================================
# PART 9 - ARENAS: from pairwise battles to one ranking.
# ========================================================================
# ONE ELO UPDATE (A beats B, both start at 1000, K=32):
#   E_A = 1/(1+10^((Rb-Ra)/400)) = 1/(1+10^0) = 0.50
#   A won, so S_A = 1.  R_A' = 1000 + 32*(1 - 0.50) = 1016.0
#            S_B = 0.  R_B' = 1000 + 32*(0 - 0.50) = 984.0
#   Equal ratings -> a coin-flip prediction -> the win moves 16 points.
#
# ------------------------------------------------------------------------
# ELO OVER ALL 18 BATTLES (start 1000, K=32).
# Same battles, two orders:
# ------------------------------------------------------------------------
#   as-listed order:  A=1083.7  B=1034.8  C=962.2  D=919.2
#   shuffled order:   A=1087.2  B=1038.8  C=961.4  D=912.6
#   difference:       A=+3.5  B=+4.0  C=-0.8  D=-6.6
#   ranking (as-listed): A > B > C > D
#   ranking (shuffled):  A > B > C > D
#   -> Same battles, different NUMBERS: Elo depends on match order.
#
# ------------------------------------------------------------------------
# BRADLEY-TERRY MLE on the aggregate win matrix (iterated):
# ------------------------------------------------------------------------
#   win matrix (row beat col):   A  B  C  D
#                           A  [ 0  2  3  3 ]
#                           B  [ 1  0  2  3 ]
#                           C  [ 0  1  0  2 ]
#                           D  [ 0  0  1  0 ]
#
#   strengths p (normalized to sum 1), by iteration:
#                     A       B       C       D
#      start      0.2500  0.2500  0.2500  0.2500
#       it 1      0.4444  0.3333  0.1667  0.0556
#       it 2      0.5195  0.3274  0.1179  0.0352
#       it 3      0.5655  0.3122  0.0933  0.0290
#      it 20      0.7015  0.2314  0.0506  0.0164
#
#   converged ranking: A > B > C > D
#   P(A beats D) = pA/(pA+pD) = 0.977
#   P(B beats C) = pB/(pB+pC) = 0.821
#   -> No order anywhere: BT reads only the totals, so its ranking is
#      the same no matter how the battles were shuffled.
# ========================================================================
