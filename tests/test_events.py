"""Typed event bus (P9-T01)."""

from __future__ import annotations

import asyncio
import threading

from cryoguard.events import EventBus, PipelineEvent


def test_sequence_numbers_are_monotonic_and_history_keeps_order() -> None:
    bus = EventBus(history=8)

    first = bus.publish("RUN_STARTED", 0.0, scenario="slow_leak_qd4")
    second = bus.publish("FRAME_TICK", 0.2, frame=3)
    third = bus.publish("RUN_FINISHED", 1.0, outcome="no_incident")

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]
    assert [event.type for event in bus.snapshot()] == [
        "RUN_STARTED",
        "FRAME_TICK",
        "RUN_FINISHED",
    ]
    assert bus.seq == 3


def test_subscribe_replays_history_and_then_receives_live_events() -> None:
    bus = EventBus()
    bus.publish("RUN_STARTED", 0.0, scenario="slow_leak_qd4")
    bus.publish("FRAME_TICK", 0.2, frame=1)

    subscription = bus.subscribe(replay=True)
    assert subscription.queue.qsize() == 2

    live = bus.publish("DECISION_CHANGE", 0.3, label="SUSPECT_LEAK", p=0.61)
    assert subscription.queue.qsize() == 3

    replayed = [subscription.queue.get_nowait() for _ in range(3)]
    assert [event.type for event in replayed] == [
        "RUN_STARTED",
        "FRAME_TICK",
        "DECISION_CHANGE",
    ]
    assert replayed[-1] is live
    bus.unsubscribe(subscription)
    assert subscription.queue.empty()


def test_replay_keeps_the_latest_events_when_history_exceeds_the_queue() -> None:
    bus = EventBus(history=10, queue_size=3)
    for index in range(6):
        bus.publish("FRAME_TICK", index / 10.0, frame=index)

    subscription = bus.subscribe(replay=True)

    frames = [subscription.queue.get_nowait().data["frame"] for _ in range(3)]
    assert frames == [3, 4, 5]


def test_a_full_queue_drops_the_oldest_event_and_never_blocks_the_producer() -> None:
    bus = EventBus(history=8, queue_size=2)
    subscription = bus.subscribe(replay=False)

    for index in range(5):
        bus.publish("FRAME_TICK", index / 10.0, frame=index)

    assert subscription.queue.qsize() == 2
    remaining = [subscription.queue.get_nowait().data["frame"] for _ in range(2)]
    assert remaining == [3, 4]
    assert [event.data["frame"] for event in bus.snapshot()] == [0, 1, 2, 3, 4]


def test_publish_from_a_worker_thread_is_delivered_by_the_bound_loop() -> None:
    async def case() -> PipelineEvent:
        bus = EventBus()
        bus.bind(asyncio.get_running_loop())
        subscription = bus.subscribe(replay=False)

        def worker() -> None:
            bus.publish("ACTION_EXECUTED", 7.6, seq=1, command="VALVE_CLOSE")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        return await asyncio.wait_for(subscription.queue.get(), timeout=2.0)

    event = asyncio.run(case())

    assert event.type == "ACTION_EXECUTED"
    assert event.data["command"] == "VALVE_CLOSE"
    assert event.t_s == 7.6


def test_a_broken_subscriber_cannot_raise_into_the_safety_loop() -> None:
    bus = EventBus()
    bus.subscribe(replay=False)

    def broken(_: PipelineEvent) -> None:
        raise RuntimeError("the dashboard fell over")

    bus._deliver = broken  # type: ignore[method-assign]

    event = bus.publish("ACTION_EXECUTED", 1.0, command="FORCE_OFF")

    assert event.seq == 1
    assert bus.snapshot()[-1].data["command"] == "FORCE_OFF"


def test_reset_restarts_the_sequence_and_drops_the_previous_run() -> None:
    bus = EventBus()
    bus.publish("RUN_STARTED", 0.0, scenario="glint_noise")
    bus.publish("FRAME_TICK", 0.1, frame=1)

    bus.reset()
    event = bus.publish("RUN_STARTED", 0.0, scenario="slow_leak_qd4")

    assert event.seq == 1
    assert [item.type for item in bus.snapshot()] == ["RUN_STARTED"]


def test_the_wire_shape_is_camel_case_t_s() -> None:
    event = PipelineEvent(type="FRAME_TICK", seq=7, t_s=1.23456, data={"frame": 12})

    assert event.to_json() == {
        "type": "FRAME_TICK",
        "seq": 7,
        "tS": 1.235,
        "data": {"frame": 12},
    }
