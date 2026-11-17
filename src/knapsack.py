"""
src/knapsack.py
---------------
0/1 Knapsack by dynamic programming.

Meaning in this project
  item   = ONE independent destination-trip option (estimated from the user's start)
  weight = its estimated cost, converted to budget UNITS (default 1 unit = Rs 100)
  value  = its ML suitability score
Goal: choose the subset with the largest total suitability whose total cost <= budget.
The chosen items are independent options, NOT one continuous multi-city journey.

Budget units: costs are rounded UP (ceil) and the budget is rounded DOWN (floor), so
the answer can never exceed the real budget. The price is a small loss of precision.

Complexity (n = items, W = budget / unit):
    Time  O(n * W)     Space O(n * W)  (the 'keep' table is needed to rebuild the answer)
"""
import math


def knapsack_select(costs, values, budget, unit=100):
    """
    costs  : list of costs in rupees
    values : list of suitability values (same length)
    budget : total budget in rupees
    unit   : size of one budget unit in rupees

    Returns dict: selected_indices, total_value, total_cost, remaining_budget
    """
    if len(costs) != len(values):
        raise ValueError("costs and values must have the same length")
    if unit <= 0:
        raise ValueError("unit must be positive")
    if budget < 0 or any(c < 0 for c in costs):
        raise ValueError("budget and costs must be non-negative")

    capacity = int(budget // unit)                          # floor
    weights = [math.ceil(c / unit) for c in costs]          # ceil
    n = len(costs)

    # dp[i][w] = best value using the first i items with capacity w
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wt, val = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                         # option 1: skip item i
            if wt <= w and val > 0:                         # option 2: take item i
                candidate = dp[i - 1][w - wt] + val
                if candidate > dp[i][w]:
                    dp[i][w] = candidate
                    keep[i][w] = True

    # Reconstruction: walk backwards through the keep table
    selected, w = [], capacity
    for i in range(n, 0, -1):
        if keep[i][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()

    total_cost = sum(costs[i] for i in selected)
    return {
        "selected_indices": selected,           # empty list => nothing affordable
        "total_value": dp[n][capacity],
        "total_cost": total_cost,
        "remaining_budget": budget - total_cost,
    }
