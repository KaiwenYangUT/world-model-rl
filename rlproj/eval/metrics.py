"""Evaluation metrics shared by every method (PLAN.md §3). numpy only."""

import math

import numpy as np


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion; returns (lo, hi). (0, 1) when n == 0."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def crafter_score(success_rates):
    """Crafter score from per-achievement success rates in [0, 1].

    Same formula as danijar/crafter analysis/common.py `compute_scores` (checked 2026-11-04):
    exp(mean(log(1 + percents))) - 1 with percents in [0, 100]; the result is in percent.
    """
    percents = 100 * np.asarray(success_rates, dtype=float)
    if not ((0 <= percents) & (percents <= 100)).all():
        raise ValueError("success rates must be in [0, 1]")
    return float(np.exp(np.mean(np.log(1 + percents))) - 1)


def area_under_curve(steps, values):
    """Trapezoid area under a learning curve, divided by the step range, so the result is the
    curve's mean value over [steps[0], steps[-1]]. Steps must be increasing."""
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    if len(steps) < 2 or not (np.diff(steps) > 0).all():
        raise ValueError("need at least two strictly increasing steps")
    area = float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(steps)))
    return area / float(steps[-1] - steps[0])


def steps_to_fraction(steps, values, fraction=0.5, final=None):
    """First step at which `values` reaches `fraction` of the final value (PLAN.md §3 sample
    efficiency). `final` defaults to the last value. None if final <= 0 or never reached."""
    final = values[-1] if final is None else final
    if final <= 0:
        return None
    for step, value in zip(steps, values):
        if value >= fraction * final:
            return step
    return None
