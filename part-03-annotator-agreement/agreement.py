"""
Evals from First Principles — Part 3: Do Humans Even Agree?

Your labels have noise; measure it before you trust them. Raw percent-agreement
lies, because two raters agree by chance whenever the labels are imbalanced. We
build Cohen's kappa by hand from a 2x2 (the chance-correction that collapses a
scary-high 0.90 raw agreement to a modest number), then Fleiss' kappa for three
raters, and close with a one-line note on Krippendorff's alpha as the general
case. Pure Python, offline, deterministic — no imports, no randomness.
"""

# --- The task -------------------------------------------------------------
# Two human annotators grade the SAME 20 support tickets: "urgent" (1) or
# "not urgent" (0). rater_A[i] and rater_B[i] describe the same ticket i.
# Most tickets are calm, so both raters say "not urgent" most of the time —
# which is exactly what inflates raw agreement.
RATER_A = [1, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RATER_B = [1, 1, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def cohen_cells(a_labels, b_labels):
    """Fill the 2x2 of two raters by hand: (both1, a1b0, a0b1, both0)."""
    both1 = a1b0 = a0b1 = both0 = 0
    for a, b in zip(a_labels, b_labels):
        if a == 1 and b == 1:
            both1 += 1
        elif a == 1 and b == 0:
            a1b0 += 1
        elif a == 0 and b == 1:
            a0b1 += 1
        else:  # a == 0 and b == 0
            both0 += 1
    return both1, a1b0, a0b1, both0


def cohen_kappa(both1, a1b0, a0b1, both0):
    """(po - pe) / (1 - pe): observed agreement corrected for chance."""
    n = both1 + a1b0 + a0b1 + both0
    po = (both1 + both0) / n
    # marginal chance each rater says "urgent" / "not urgent"
    a_urg = (both1 + a1b0) / n
    b_urg = (both1 + a0b1) / n
    a_not, b_not = 1 - a_urg, 1 - b_urg
    pe = a_urg * b_urg + a_not * b_not
    kappa = (po - pe) / (1 - pe)
    return po, pe, kappa, (a_urg, b_urg, a_not, b_not)


def fleiss_kappa(counts, n_raters):
    """Fleiss' kappa from a per-item category-count table (any # of raters).

    counts[i] = [n_cat0, n_cat1] = how many raters put item i in each category.
    Returns (P_bar, P_e, kappa, item_agreements, category_props).
    """
    n_items = len(counts)
    # P_i: fraction of rater PAIRS on item i that agree
    item_agree = []
    for row in counts:
        s = sum(c * c for c in row)
        item_agree.append((s - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(item_agree) / n_items
    # p_j: overall share of assignments landing in each category
    total = n_items * n_raters
    cat_props = [sum(row[j] for row in counts) / total for j in range(len(counts[0]))]
    p_e = sum(p * p for p in cat_props)
    kappa = (p_bar - p_e) / (1 - p_e)
    return p_bar, p_e, kappa, item_agree, cat_props


if __name__ == "__main__":
    line = "=" * 72
    print(line)
    print("PART 3 - DO HUMANS EVEN AGREE? chance-corrected agreement by hand.")
    print(line)

    # --- Cohen's kappa: two raters, 20 tickets ----------------------------
    print("TWO RATERS, 20 TICKETS: each labels 'urgent' (1) or 'not urgent' (0).")
    print()
    print("  item  A  B  agree")
    for i, (a, b) in enumerate(zip(RATER_A, RATER_B), start=1):
        print(f"  {i:>4}  {a}  {b}  {'yes' if a == b else 'NO '}")

    both1, a1b0, a0b1, both0 = cohen_cells(RATER_A, RATER_B)
    n = len(RATER_A)
    agree = both1 + both0
    print()
    print("  2x2 of A vs B (counted by hand):")
    print(f"                 B=urgent   B=not")
    print(f"    A=urgent  {both1:>7}   {a1b0:>5}")
    print(f"    A=not     {a0b1:>7}   {both0:>5}")

    po, pe, kappa, (a_urg, b_urg, a_not, b_not) = cohen_kappa(both1, a1b0, a0b1, both0)
    print()
    print(f"  raw agreement po = (both-urgent + both-not)/n = ({both1}+{both0})/{n} = {po:.2f}")
    print(f"  ...a scary-high {po:.0%}. But how much of that is just chance?")
    print()
    print(f"  each rater's marginal rates:")
    print(f"    A says urgent {both1 + a1b0}/{n} = {a_urg:.2f},  not {a_not:.2f}")
    print(f"    B says urgent {both1 + a0b1}/{n} = {b_urg:.2f},  not {b_not:.2f}")
    print(f"  chance agreement pe = {a_urg:.2f}*{b_urg:.2f} + {a_not:.2f}*{b_not:.2f}")
    print(f"                      = {a_urg * b_urg:.4f} + {a_not * b_not:.4f} = {pe:.4f}")
    print()
    print(f"  Cohen kappa = (po - pe)/(1 - pe)")
    print(f"              = ({po:.2f} - {pe:.4f})/(1 - {pe:.4f}) = {kappa:.2f}")
    print(f"  -> {po:.0%} raw agreement is really kappa {kappa:.2f}: two raters who")
    print(f"     almost always say 'not urgent' agree by chance {pe:.0%} of the time,")
    print(f"     so the honest, chance-corrected number is far lower.")

    # --- Fleiss' kappa: three raters --------------------------------------
    print()
    print("-" * 72)
    print("THREE RATERS: Cohen only handles two, so use Fleiss' kappa.")
    print("Five fresh tickets, each graded by raters R1, R2, R3 (1=urgent, 0=not).")
    print("-" * 72)
    R1 = [1, 1, 0, 1, 0]
    R2 = [1, 1, 0, 0, 0]
    R3 = [1, 0, 0, 0, 0]
    N_RATERS = 3
    print()
    print("  item  R1 R2 R3  | urgent  not")
    counts = []
    for i in range(len(R1)):
        votes = [R1[i], R2[i], R3[i]]
        n_urg = sum(votes)
        n_not = N_RATERS - n_urg
        counts.append([n_not, n_urg])  # [category 0 = not, category 1 = urgent]
        print(f"  {i + 1:>4}   {R1[i]}  {R2[i]}  {R3[i]}  |{n_urg:>6}  {n_not:>4}")

    p_bar, p_e, fk, item_agree, cat_props = fleiss_kappa(counts, N_RATERS)
    print()
    print("  per-item agreement P_i = (sum of squared vote-counts - raters)")
    print("                           / (raters * (raters - 1)):")
    for i, pi in enumerate(item_agree, start=1):
        print(f"    item {i}: P_i = {pi:.4f}")
    print(f"  mean observed agreement  P_bar = {p_bar:.4f}")
    p_not, p_urg = cat_props
    print(f"  category shares: urgent {p_urg:.2f}, not {p_not:.2f}")
    print(f"  chance agreement P_e = {p_urg:.2f}^2 + {p_not:.2f}^2 = {p_e:.2f}")
    print(f"  Fleiss kappa = (P_bar - P_e)/(1 - P_e)")
    print(f"               = ({p_bar:.4f} - {p_e:.2f})/(1 - {p_e:.2f}) = {fk:.2f}")

    # --- Krippendorff's alpha: the general case ---------------------------
    print()
    print("-" * 72)
    print("Krippendorff's alpha is the general case: any number of raters,")
    print("missing labels, and any scale (nominal / ordinal / interval). It is")
    print("built from disagreement instead of agreement, but the idea is the")
    print("same one we just did by hand: correct the raw number for chance.")
    print(line)

# Expected output:
# ========================================================================
# PART 3 - DO HUMANS EVEN AGREE? chance-corrected agreement by hand.
# ========================================================================
# TWO RATERS, 20 TICKETS: each labels 'urgent' (1) or 'not urgent' (0).
#
#   item  A  B  agree
#      1  1  1  yes
#      2  1  1  yes
#      3  1  0  NO 
#      4  0  1  NO 
#      5  0  0  yes
#      6  0  0  yes
#      7  0  0  yes
#      8  0  0  yes
#      9  0  0  yes
#     10  0  0  yes
#     11  0  0  yes
#     12  0  0  yes
#     13  0  0  yes
#     14  0  0  yes
#     15  0  0  yes
#     16  0  0  yes
#     17  0  0  yes
#     18  0  0  yes
#     19  0  0  yes
#     20  0  0  yes
#
#   2x2 of A vs B (counted by hand):
#                  B=urgent   B=not
#     A=urgent        2       1
#     A=not           1      16
#
#   raw agreement po = (both-urgent + both-not)/n = (2+16)/20 = 0.90
#   ...a scary-high 90%. But how much of that is just chance?
#
#   each rater's marginal rates:
#     A says urgent 3/20 = 0.15,  not 0.85
#     B says urgent 3/20 = 0.15,  not 0.85
#   chance agreement pe = 0.15*0.15 + 0.85*0.85
#                       = 0.0225 + 0.7225 = 0.7450
#
#   Cohen kappa = (po - pe)/(1 - pe)
#               = (0.90 - 0.7450)/(1 - 0.7450) = 0.61
#   -> 90% raw agreement is really kappa 0.61: two raters who
#      almost always say 'not urgent' agree by chance 74% of the time,
#      so the honest, chance-corrected number is far lower.
#
# ------------------------------------------------------------------------
# THREE RATERS: Cohen only handles two, so use Fleiss' kappa.
# Five fresh tickets, each graded by raters R1, R2, R3 (1=urgent, 0=not).
# ------------------------------------------------------------------------
#
#   item  R1 R2 R3  | urgent  not
#      1   1  1  1  |     3     0
#      2   1  1  0  |     2     1
#      3   0  0  0  |     0     3
#      4   1  0  0  |     1     2
#      5   0  0  0  |     0     3
#
#   per-item agreement P_i = (sum of squared vote-counts - raters)
#                            / (raters * (raters - 1)):
#     item 1: P_i = 1.0000
#     item 2: P_i = 0.3333
#     item 3: P_i = 1.0000
#     item 4: P_i = 0.3333
#     item 5: P_i = 1.0000
#   mean observed agreement  P_bar = 0.7333
#   category shares: urgent 0.40, not 0.60
#   chance agreement P_e = 0.40^2 + 0.60^2 = 0.52
#   Fleiss kappa = (P_bar - P_e)/(1 - P_e)
#                = (0.7333 - 0.52)/(1 - 0.52) = 0.44
#
# ------------------------------------------------------------------------
# Krippendorff's alpha is the general case: any number of raters,
# missing labels, and any scale (nominal / ordinal / interval). It is
# built from disagreement instead of agreement, but the idea is the
# same one we just did by hand: correct the raw number for chance.
# ========================================================================
