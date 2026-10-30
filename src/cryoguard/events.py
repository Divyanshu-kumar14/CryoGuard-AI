"""Typed pipeline event bus (P9-T01).

The headless CLI and the dashboard must observe the *same* sequence of facts.
This module is the single place that sequence is defined; the pipeline publishes
to it, the server forwards it, and the UI renders it. Nothing here can influence
a decision: ``publish`` never raises into the caller, and a slow or absent
subscriber only loses its own copy of an event.

Five event types are public (the spec names them):

``FRAME_TICK``       one classical tick: candidate contours, camera health
``EVIDENCE_UPDATE``  the fusion inputs for that tick
``DECISION_CHANGE``  the gated decision; ``changed`` flags a band/rule change
``ACTION_EXECUTED``  one orchestrator step, emitted as it is appended
``INCIDENT_CLOSED``  the sealed report

``RUN_STARTED`` and ``RUN_FINISHED`` bracket a run so a late subscriber can tell
a finished run from an idle server.

Threading: the confirmed-leak sequence runs in a worker thread (the Modbus and
Redfish clients are blocking), so ``publish`` may be called from any thread. If
the bus is bound to a loop, delivery is marshalled with
``loop.call_soon_threadsafe``; otherwise it is delivered inline. Each subscriber
owns an ``asyncio.Queue`` and a full queue drops its oldest event rather than
blocking the producer.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

EventType = Literal[
    "RUN_STARTED",
    "FRAME_TICK",
    "EVIDENCE_UPDATE",
    "DECISION_CHANGE",
    "ACTION_EXECUTED",
    "INCIDENT_CLOSED",
    "RUN_FINISHED",
]


@dataclass(frozen=True)
class PipelineEvent:
    """One fact on the wire. ``data`` is JSON-shaped by construction."""

    type: EventType
    seq: int
    t_s: float
    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "seq": self.seq,
            "tS": round(float(self.t_s), 3),
            "data": self.data,
        }


@dataclass
class Subscription:
    """A subscriber's private queue. The id is only used to unsubscribe."""

    id: int
    queue: asyncio.Queue[PipelineEvent]


class EventBus:
    def __init__(self, *, history: int = 4096, queue_size: int = 1024) -> None:
        if history < 1:
            raise ValueError(f"history must be positive: {history}")
        if queue_size < 1:
            raise ValueError(f"queue_size must be positive: {queue_size}")
        self._history: deque[PipelineEvent] = deque(maxlen=history)
        self._queue_size = queue_size
        self._listeners: dict[int, asyncio.Queue[PipelineEvent]] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._next_id = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind delivery to the serving loop; safe to call once per run."""

        self._loop = loop

    def reset(self) -> None:
        """Start a new run: sequence numbers restart and old history is dropped.

        Subscribers stay attached; they are about to receive the new run. This
        is deliberate: a dashboard connected across runs should not replay the
        previous incident as if it were current.
        """

        with self._lock:
            self._history.clear()
            self._seq = 0

    def subscribe(self, *, replay: bool = True) -> Subscription:
        """Register a listener. History and registration happen under one lock.

        Doing both atomically is what makes replay race-free: an event is either
        in the copied history or delivered to the queue, never both.
        """

        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            subscription = Subscription(id=self._next_id, queue=queue)
            self._next_id += 1
            self._listeners[subscription.id] = queue
            if replay:
                # A late subscriber needs the latest state, not the oldest events.
                for event in list(self._history)[-self._queue_size :]:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        break
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._listeners.pop(subscription.id, None)

    def snapshot(self) -> tuple[PipelineEvent, ...]:
        """The events of the current run, oldest first. Detached copy."""

        with self._lock:
            return tuple(self._history)

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, event_type: EventType, t_s: float, **data: Any) -> PipelineEvent:
        """Record and deliver one event. This function never raises.

        A dashboard is not in the safety loop (NFR-SAF-01). If a subscriber
        queue misbehaves, the event is still in history and the sequence keeps
        running; the alternative would let a browser tab stop a valve close.
        """

        with self._lock:
            self._seq += 1
            event = PipelineEvent(type=event_type, seq=self._seq, t_s=float(t_s), data=dict(data))
            self._history.append(event)
            loop = self._loop

        try:
            if loop is not None:
                if loop.is_running():
                    loop.call_soon_threadsafe(self._deliver, event)
                # A closing loop gets the history but no cross-thread queue write.
            else:
                self._deliver(event)
        except Exception:
            # Delivery is best-effort; the recorded event is the source of truth.
            pass
        return event

    def _deliver(self, event: PipelineEvent) -> None:
        with self._lock:
            queues = list(self._listeners.values())
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:
                # One broken subscriber must not starve the others.
                continue
