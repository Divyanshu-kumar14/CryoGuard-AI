"""Clopper-Pearson and the small-sample helpers (P8-T02)."""

from __future__ import annotations

import pytest

from cryoguard.bench.metrics import (
    MetricsError,
    clopper_pearson,
    mean_std,
    percentile,
    regularized_beta,
)


def test_ten_out_of_ten_matches_the_closed_form() -> None:
    """Lower = 0.025**(1/10) = 0.6915, upper = 1."""

    low, high = clopper_pearson(10, 10)

    assert low == pytest.approx(0.025 ** (1 / 10), abs=1e-9)
    assert low == pytest.approx(0.6915, abs=5e-4)
    assert high == 1.0


def test_zero_out_of_fifteen_matches_the_closed_form() -> None:
    low, high = clopper_pearson(0, 15)

    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 15), abs=1e-9)
    assert high == pytest.approx(0.2180, abs=5e-4)


def test_twelve_out_of_twelve() -> None:
    low, high = clopper_pearson(12, 12)

    assert low == pytest.approx(0.7354, abs=5e-4)
    assert high == 1.0


def test_five_out_of_ten_matches_the_published_interval() -> None:
    low, high = clopper_pearson(5, 10)

    assert low == pytest.approx(0.1871, abs=5e-4)
    assert high == pytest.approx(0.8129, abs=5e-4)


def test_the_interval_containing_the_point_estimate() -> None:
    for k, n in ((0, 1), (1, 2), (3, 7), (9, 10), (10, 10)):
        low, high = clopper_pearson(k, n)
        assert 0.0 <= low <= k / n <= high <= 1.0


def test_no_trials_has_no_interval() -> None:
    assert clopper_pearson(0, 0) is None


def test_impossible_counts_are_refused() -> None:
    with pytest.raises(MetricsError, match="impossible"):
        clopper_pearson(3, 2)
    with pytest.raises(MetricsError, match="non-negative"):
        clopper_pearson(0, -1)
    with pytest.raises(MetricsError, match="confidence"):
        clopper_pearson(1, 2, 1.0)


def test_regularized_beta_endpoints() -> None:
    assert regularized_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_beta(2.0, 3.0, 1.0) == 1.0
    assert regularized_beta(1.0, 1.0, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_percentiles_are_interpolated_and_empty_is_none() -> None:
    assert percentile([], 0.5) is None
    assert percentile([4.0], 0.95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_mean_and_sample_std() -> None:
    mean, std = mean_std([7.6, 3.5])

    assert mean == pytest.approx(5.55)
    assert std == pytest.approx(2.8991, abs=1e-3)
    assert mean_std([]) == (None, None)
    assert mean_std([2.0]) == (2.0, None)
