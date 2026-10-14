"""S87-m07: the finalize validation pass must not starve the event loop.

Regression for the CAT-R001-S87 lease-loss incident: the whole-document
critique ran synchronously on the event-loop thread, freezing the lease
heartbeat coroutine for minutes so the 120s lease expired and the completed
report was rejected on write (mission BLOCKED). The critique must run off the
loop (asyncio.to_thread), like the synthesis already does.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import deepsearch.agent.codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.critique import CritiqueResult


async def _fake_liveness(**kwargs: object) -> tuple[list, LivenessTelemetry]:
    return (list(kwargs.get("annotations") or []), LivenessTelemetry())


def test_validation_pass_critique_does_not_starve_event_loop() -> None:
    sleep_seconds = 1.0
    tick = 0.02

    def slow_critique(*args: object, **kwargs: object) -> CritiqueResult:
        # Stand-in for the minutes-long blocking whole-document critique LLM call.
        time.sleep(sleep_seconds)
        return CritiqueResult(
            output_markdown="# Report\n\nok.",
            claim_count=1,
            gap_notes_inserted=0,
            runtime_seconds=sleep_seconds,
        )

    async def run() -> tuple[int, object]:
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(tick)

        ticker_task = asyncio.create_task(ticker())
        # let the ticker establish a baseline cadence
        await asyncio.sleep(tick * 2)
        with patch.object(
            fnode,
            "_extract_and_resolve_citations",
            return_value=("# Report\n\nok.", []),
        ), patch.object(
            fnode, "_run_liveness_check", _fake_liveness
        ), patch.object(
            fnode, "run_critique", side_effect=slow_critique
        ):
            result = await fnode._run_validation_pass(
                mission_id="M",
                draft_markdown="# Report\n\nok.",
                sources_collected=[],
                mission_context={},
            )
        stop.set()
        await ticker_task
        return ticks, result

    ticks, result = asyncio.run(run())

    # A free loop yields ~sleep_seconds/tick (~50) ticks during the critique.
    # If the critique blocks the loop (the incident bug), the ticker stalls to
    # ~0 increments across that window.
    assert ticks >= 20, f"event loop starved during critique (ticks={ticks})"
    assert result.critique_result.claim_count == 1
