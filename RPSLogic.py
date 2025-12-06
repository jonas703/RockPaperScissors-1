# ---------------------------------------------------------
# Rock-Paper-Scissors Logic Engine
# ---------------------------------------------------------

RPSM1 = [['RR', 'PR', 'SR'], ['RP', 'PP', 'SP'], ['RS', 'PS', 'SS']]
RPS = ['R', 'P', 'S']
WorL_matrix = [
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
]


# ---------------------------------------------------------
# Core engine: build list of (p1, p2, outcome)
# ---------------------------------------------------------
def build_results(p1_choices, p2_choices):
    results = []
    for p1 in p1_choices:
        for p2 in p2_choices:
            p1_index = RPS.index(p1)
            p2_index = RPS.index(p2)
            outcome = WorL_matrix[p2_index][p1_index]  # 1 win, -1 loss, 0 draw
            results.append((p1, p2, outcome))
    return results


# ---------------------------------------------------------
# Main function to determine the best move to drop
# ---------------------------------------------------------
def RPSMinus1(player1_choices, player2_choices, strategy=3):

    results = build_results(player1_choices, player2_choices)

    losing = [r for r in results if r[2] == -1]
    winning = [r for r in results if r[2] == 1]
    drawing = [r for r in results if r[2] == 0]

    # 1. Auto-win condition
    if len(winning) == len(results):
        return {
            "status": "win",
            "message": "ALL ROADS LEAD TO VICTORY!",
            "recommended_drop": None,
            "results": results
        }

    # 2. Identify unique losing moves
    losing_moves = [r[0] for r in losing]
    unique_losing_moves = list(set(losing_moves))

    # 3. Baseline valuation
    base_value = {m: 0 for m in player1_choices}
    for p1, _, out in winning:
        base_value[p1] += 1
    for p1, _, out in losing:
        base_value[p1] -= 1

    # -----------------------------------------
    # Strategy-weighted scoring
    # -----------------------------------------
    # values for different strategies, 1: Aggressive, 2: Defensive, 3: Balanced, 4: Default
    values = {
        1: [2, 1, -0.5, 1.5, 1, -0.5],
        2: [1, -1.5, 0.5, 1, 1, 0.5],
        3: [1.5, 1, 0, 1, 0.5, 0],
        4: [1, 1, 0, 1, 1, 0]
    }

    weighted = {m: 0 for m in player1_choices}

    # predict likely P2 move
    p2_val = {m: 0 for m in player2_choices}
    for _, p2, out in winning:
        p2_val[p2] -= 1
    for _, p2, out in losing:
        p2_val[p2] += 1
    p2_likely = max(p2_val, key=p2_val.get)

    # scoring loop
    for p1, p2, outcome in results:
        score = 0
        use_values = values[strategy]

        if p2 == p2_likely:     # same-move weighting
            if outcome == 1: score += use_values[0]
            elif outcome == -1: score -= use_values[1]
            else: score += use_values[2]
        else:                  # other-move weighting
            if outcome == 1: score += use_values[3]
            elif outcome == -1: score -= use_values[4]
            else: score += use_values[5]

        weighted[p1] += score

    # -----------------------------------------
    # Compute recommended drop
    # -----------------------------------------
    recommended_drop = min(weighted, key=weighted.get)

    return {
        "status": "ok",
        "results": results,
        "losing": losing,
        "winning": winning,
        "drawing": drawing,
        "base_value": base_value,
        "weighted_value": weighted,
        "likely_p2_move": p2_likely,
        "recommended_drop": recommended_drop,
        "message": f"DROP {recommended_drop} to minimize loss."
    }
