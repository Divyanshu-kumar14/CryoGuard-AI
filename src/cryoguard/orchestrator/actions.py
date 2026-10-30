"""Append-only action log (P6-T08).

``deterministicActionsTaken`` is the evidence of what the machine did, so it is
built by one writer, in execution order, with non-decreasing timestamps, and it
is sealed the moment the incident closes. After ``seal()`` an append raises
``IncidentClosed``; the report layer only ever receives ``snapshot()``, a tuple
of frozen records that cannot be edited in place.

This module deliberately has no dependency on ``cryoguard.report``. The
orchestrator produces the facts; the composition root turns them into a report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias


class Clock(Protocol):
    def now(self) -> float: ...
    def advance(self, seconds: float) -> float: ...


StepCallback: TypeAlias = Callable[["SequenceStep", int], None]


class IncidentClosed(RuntimeError):
    """A closed incident cannot grow or change."""


class ActionOrderError(RuntimeError):
    """A step timestamped before its predecessor is not an ordered log."""


@dataclass(frozen=True)
class SequenceStep:
    """One executed action, in execution order, for the read-only incident log."""

    ts: float
    actor: str
    command: str
    status: str
    detail: str = ""


class ActionLog:
    def __init__(self, clock: Clock, on_step: StepCallback | None = None) -> None:
        self._clock = clock
        self._steps: list[SequenceStep] = []
        self._sealed = False
        self._on_step = on_step

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def steps(self) -> tuple[SequenceStep, ...]:
        return tuple(self._steps)

    def snapshot(self) -> tuple[SequenceStep, ...]:
        """The only view handed to the report service: immutable and detached."""

        return self.steps

    def append(self, actor: str, command: str, status: str, detail: str = "") -> SequenceStep:
        if self._sealed:
            raise IncidentClosed(f"incident closed; refused to append {command} ({actor})")
        if not command.strip():
            raise ValueError("an action needs a command")
        ts = self._clock.now()
        if self._steps and ts < self._steps[-1].ts:
            raise ActionOrderError(
                f"{command} at t={ts} precedes {self._steps[-1].command} at t={self._steps[-1].ts}"
            )
        step = SequenceStep(ts=ts, actor=actor, command=command, status=status, detail=detail)
        self._steps.append(step)
        if self._on_step is not None:
            # The observer runs after the step is durable in the log. It must not
            # be able to veto or reorder an actuation; EventBus.publish swallows
            # delivery failures for exactly this reason.
            self._on_step(step, len(self._steps))
        return step

    def seal(self) -> tuple[SequenceStep, ...]:
        self._sealed = True
        return self.steps
