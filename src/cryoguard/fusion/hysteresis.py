"""Temporal gating around the stateless rule table.

The rules say what the evidence means right now. This class says how long it has
meant it.

Two independent counters, and they must not be conflated:

* ``_streak`` counts consecutive ticks in which the *instantaneous* band matches
  the band currently being reported. The rule table reads this count, so a leak
  still needs three consecutive high ticks even if a single calm tick did not
  de-escalate the state.
* ``_drop_count`` counts consecutive ticks in which the score is below the held
  band's floor minus the de-escalation margin. Only after
  ``decision.deescalation_ticks`` such ticks does the state step down (PRD §6.4).

Incident labels latch: ``CONFIRMED_LEAK`` and ``EMERGENCY_TRIP`` never walk back
without an operator ``reset()``. ``CAMERA_DEGRADED`` is deliberately exempt from
the hold in both directions - a blind camera is not a suspicion band, and a
recovered camera must not keep reporting a degraded one for five ticks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from cryoguard.fusion.decision import decide
from cryoguard.models import Decision, Evidence, SystemLabel


class HysteresisDecision:
    LATCHED = {SystemLabel.CONFIRMED_LEAK, SystemLabel.EMERGENCY_TRIP}

    # Severity order used only to answer "is this tick a step down?". R4 is an
    # incident and outranks a high-score visual anomaly, so it is never delayed
    # behind one. R1/R1b latch on arrival and never reach the comparison.
    RANK = {"R7": 0, "R6": 1, "R5": 2, "R3": 3, "HIGH": 4, "R4": 5, "R1": 6, "R1b": 6}

    # The threshold whose -margin deadband governs de-escalation out of a band.
    # R4 is entered on hydraulics rather than a score; the suspect cutoff is the
    # floor that means "no longer a suspect" for it.
    FLOOR_KEYS = {
        "HIGH": "confirmed_leak",
        "R4": "suspect_cutoff",
        "R3": "escalation_cutoff",
        "R5": "suspect_cutoff",
        "R6": "nominal_cutoff",
    }

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._key: str | None = None
        self._held: Decision | None = None
        self._streak = 0
        self._drop_count = 0
        self._latched: Decision | None = None

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def drop_streak(self) -> int:
        return self._drop_count

    @property
    def latched(self) -> Decision | None:
        return self._latched

    def reset(self) -> None:
        self._key = None
        self._held = None
        self._streak = 0
        self._drop_count = 0
        self._latched = None

    @staticmethod
    def _band(rule_id: str) -> str:
        """R2/R2b/R2c are the same evidence band at different completeness."""

        if rule_id in {"R2", "R2b", "R2c"}:
            return "HIGH"
        return rule_id

    @staticmethod
    def _rank(band: str) -> int:
        try:
            return HysteresisDecision.RANK[band]
        except KeyError as exc:  # a new rule without a severity is a bug, not a guess
            raise ValueError(f"no severity rank for evidence band {band!r}") from exc

    def _adopt(self, decision: Decision, band: str) -> Decision:
        self._key = band
        self._held = decision
        self._streak = 1
        self._drop_count = 0
        if decision.label in self.LATCHED:
            self._latched = decision
        return decision

    def update(self, ev: Evidence, t_s: float) -> Decision:
        # An incident never auto-de-escalates (PRD §6.4, T-DEC-03). The latch is
        # explicit state, not a consequence of the pipeline returning early, so a
        # future streaming loop cannot quietly walk a confirmed leak back to
        # NOMINAL. Only an operator reset() clears it.
        if self._latched is not None:
            return self._latched

        instantaneous = decide(replace(ev, ticks=1), self.cfg)
        band = self._band(instantaneous.rule_id)

        # A hard fault is not a band to hold, and neither is a camera failure.
        if self._held is None or band == "R8" or self._key == "R8":
            return self._adopt(instantaneous, band)

        assert self._key is not None  # narrowed by the branch above
        if band == self._key:
            self._streak += 1
            self._drop_count = 0
            decision = decide(replace(ev, ticks=self._streak), self.cfg)
            self._held = decision
            if decision.label in self.LATCHED:
                self._latched = decision
            return decision

        if self._rank(band) > self._rank(self._key):
            return self._adopt(instantaneous, band)

        # A lower band is a candidate, not a transition. The state only steps
        # down after the score has sat below the held band's floor minus the
        # margin for the configured number of ticks.
        floor = self.cfg["fusion"]["thresholds"][self.FLOOR_KEYS[self._key]]
        margin = float(self.cfg["fusion"]["thresholds"]["deescalation_margin"])
        if instantaneous.p < float(floor) - margin:
            self._drop_count += 1
        else:
            self._drop_count = 0

        # The current tick's evidence does not satisfy the held band, so it does
        # not extend the escalation streak either.
        self._streak = 0

        required = int(self.cfg["decision"]["deescalation_ticks"])
        if required <= 0:
            return self._adopt(instantaneous, band)
        if self._drop_count >= required:
            return self._adopt(instantaneous, band)

        self._held = replace(self._held, p=instantaneous.p)
        return self._held
