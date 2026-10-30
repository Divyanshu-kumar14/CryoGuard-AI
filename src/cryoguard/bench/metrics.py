"""Clopper-Pearson intervals and small-sample statistics (P8-T02).

No scipy: the exact interval needs two Beta quantiles, and the regularized
incomplete beta with a continued fraction plus a bisection is short enough to
own, deterministic, and has no install cost for the judges. The known-value
tests pin it against closed forms (10/10 lower = 0.025**(1/10)).
"""

from __future__ import annotations

import math


class MetricsError(ValueError):
    """A statistic was asked for outside its domain."""


def _betacf(a: float, b: float, x: float, *, max_iter: int = 200) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""

    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break

    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function."""

    if not 0.0 <= x <= 1.0:
        raise MetricsError(f"x must be in [0, 1], got {x}")
    if a <= 0 or b <= 0:
        raise MetricsError(f"a and b must be positive, got {a}, {b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, *, iterations: int = 200) -> float:
    """Inverse regularized beta by bisection; monotone and deterministic."""

    if not 0.0 < q < 1.0:
        raise MetricsError(f"q must be in (0, 1), got {q}")
    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        if regularized_beta(a, b, middle) < q:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Exact two-sided interval. ``None`` when there is nothing to estimate."""

    if trials < 0:
        raise MetricsError(f"trials must be non-negative: {trials}")
    if not 0 <= successes <= trials:
        raise MetricsError(f"{successes} successes out of {trials} is impossible")
    if not 0.0 < confidence < 1.0:
        raise MetricsError(f"confidence must be in (0, 1): {confidence}")
    if trials == 0:
        return None

    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else beta_ppf(alpha / 2.0, successes, trials - successes + 1)
    high = (
        1.0
        if successes == trials
        else beta_ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return (low, high)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty sample."""

    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise MetricsError(f"q must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Sample mean and sample standard deviation (n-1). ``None`` when n=0."""

    if not values:
        return (None, None)
    count = len(values)
    mean = sum(values) / count
    if count == 1:
        return (mean, None)
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return (mean, math.sqrt(variance))
