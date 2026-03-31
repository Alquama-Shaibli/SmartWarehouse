def grade(env):
    sm = env.state_manager

    score = 0

    # completion
    completed_orders = 0
    for order in sm.orders:
        if all(item in sm.carrying for item in order["items"]):
            completed_orders += 1

    score += completed_orders / len(sm.orders) * 0.5

    # efficiency
    score += max(0, 0.3 - sm.steps * 0.005)

    # safety
    score += max(0, 0.2 - sm.collisions * 0.05)

    return round(min(score, 1.0), 3)
