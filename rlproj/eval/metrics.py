"""Evaluation metrics shared by every method (PLAN.md §3). numpy only."""

import math


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion; returns (lo, hi). (0, 1) when n == 0."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)
