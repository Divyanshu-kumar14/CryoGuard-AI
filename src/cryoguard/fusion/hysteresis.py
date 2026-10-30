"""Temporal gating around the stateless rule table.

The rules say what the evidence means right now. This class says how long it has
meant it. Counters are keyed on the *instantaneous* band (the rule that would fire
with one tick of history), so a glint that appears for a single tick cannot
inherit the streak of the leak that preceded it.

De-escalation is deliberately absent: PRD §6.3 says a confirmed leak or an
emergency trip never de-escalates without an operator reset.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from cryoguard.fusion.decision import decide
from cryoguard.models import Decision, Evidence


class HysteresisDecision:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._key: tuple[str, str] | None = None
        self._count = 0
        self._first_t = 0.0

    @property
    def streak(self) -> int:
        return self._count

    def reset(self) -> None:
        self._key = None
        self._count = 0
        self._first_t = 0.0

    @staticmethod
    def _band(rule_id: str) -> str:
        """R2/R2b/R2c are the same evidence band at different completeness."""

        if rule_id in {"R2", "R2b", "R2c"}:
            return "HIGH"
        return rule_id

    def update(self, ev: Evidence, t_s: float) -> Decision:
        instantaneous = decide(replace(ev, ticks=1), self.cfg)
        key = (self._band(instantaneous.rule_id), instantaneous.label.value)

        if key != self._key:
            self._key = key
            self._count = 1
            self._first_t = t_s
        else:
            self._count += 1

        return decide(replace(ev, ticks=self._count), self.cfg)
