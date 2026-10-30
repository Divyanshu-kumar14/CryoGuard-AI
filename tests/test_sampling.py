"""Adaptive tick policy (D-18)."""

from __future__ import annotations

import pytest

from cryoguard.models import SystemLabel
from cryoguard.perception.sampling import SamplingPolicy


def test_idle_tracking_and_suspect_rates() -> None:
    policy = SamplingPolicy({"idle_hz": 2, "track_hz": 5, "suspect_hz": 10})

    assert policy.update(None, has_candidate=False) == 2
    assert policy.update(SystemLabel.NOMINAL, has_candidate=True) == 5
    assert policy.update(SystemLabel.SUSPECT_LEAK, has_candidate=True) == 10
    assert policy.changes == 2  # the initial idle rate is not a change


def test_returning_to_nominal_backs_the_rate_down() -> None:
    policy = SamplingPolicy({"idle_hz": 2, "track_hz": 5, "suspect_hz": 10})
    policy.update(SystemLabel.UNVERIFIED_ANOMALY, has_candidate=True)
    assert policy.update(SystemLabel.NOMINAL, has_candidate=False) == 2


def test_ratings_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="idle <= track <= suspect"):
        SamplingPolicy({"idle_hz": 10, "track_hz": 5, "suspect_hz": 2})


def test_period_is_the_reciprocal() -> None:
    policy = SamplingPolicy({"idle_hz": 2, "track_hz": 5, "suspect_hz": 10})
    assert policy.period_s() == pytest.approx(0.5)
