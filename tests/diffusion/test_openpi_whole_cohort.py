# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.entrypoints.openpi.connection import RobotRealtimeConnection, _pack, _unpack
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(rid, steps=1):
    return OmniDiffusionRequest(
        prompt="observation", request_id=rid, sampling_params=OmniDiffusionSamplingParams(num_inference_steps=steps)
    )


def _scheduler(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_COHORT_SCHEMA", raising=False)
    monkeypatch.delenv("VLLM_OMNI_COHORT_SCHEDULING", raising=False)
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=4, request_batch_max_wait_ms=1000.0))
    return scheduler


def test_whole_cohort_schedules_without_waiting_or_mixing(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    waiting = scheduler.get_admission_wait_decision(now=10.0)
    assert waiting.should_wait
    scheduler.add_whole_cohort([_request("a"), _request("b")])
    scheduler.add_whole_cohort([_request("c")])
    assert not scheduler.get_admission_wait_decision(now=10.01).should_wait
    assert scheduler.should_end_admission_wait(waiting, now=10.01, stable_since=10.0)
    assert list(scheduler.schedule().scheduled_request_ids) == ["a", "b"]


def test_whole_cohort_rejects_incompatible_members_and_cancels_queued_members(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    with pytest.raises(ValueError, match="incompatible sampling parameters"):
        scheduler.add_whole_cohort([_request("a"), _request("b", steps=2)])
    assert not scheduler.has_requests()
    scheduler.add_whole_cohort([_request("a"), _request("b")])
    scheduler.add_whole_cohort([_request("c")])
    scheduler.finish_requests("a", DiffusionRequestStatus.FINISHED_ABORTED)
    assert list(scheduler.schedule().scheduled_request_ids) == ["c"]


def _members():
    return [
        {"request_id": "r-a", "session_id": "a", "session_sequence": 0},
        {"request_id": "r-b", "session_id": "b", "session_sequence": 0},
    ]


async def _serving(monkeypatch, infer):
    monkeypatch.setenv("VLLM_OMNI_OPENPI_WHOLE_COHORT", "1")
    replica = SimpleNamespace(whole_cohort_capacity=4, infer_whole_cohort=infer)
    pool = SimpleNamespace(stage_type="diffusion", live_replica_ids=lambda: (0,), stage_client=replica)
    engine = SimpleNamespace(stage_pools=(pool,), orchestrator_loop=asyncio.get_running_loop())
    serving = object.__new__(ServingRealtimeRobotOpenPI)
    serving.engine_client = SimpleNamespace(engine=engine)
    serving._cohort_session_sequences = {}
    serving.build_request = Mock(side_effect=lambda obs, **_: _request(obs["request_id"]))
    return serving


@pytest.mark.asyncio
async def test_whole_cohort_validates_before_submission_and_rejects_replay(monkeypatch):
    results = [SimpleNamespace(multimodal_output={"actions": [1]}), RuntimeError("inference failed")]
    infer = AsyncMock(return_value=results)
    serving = await _serving(monkeypatch, infer)
    a, b = _members()
    with pytest.raises(ValueError, match="identity"):
        await serving.infer_whole_cohort("c", [a, dict(b, session_id="a")])
    infer.assert_not_awaited()
    first, second = await serving.infer_whole_cohort("c", [a, b])
    assert first.tolist() == [1.0]
    assert second is None
    (requests,) = infer.await_args.args
    assert [req.request_id for req in requests] == ["r-a", "r-b"]
    assert all(req.sampling_params.extra_args["whole_cohort_id"] == "c" for req in requests)
    with pytest.raises(ValueError, match="replayed"):
        await serving.infer_whole_cohort("c", [a, b])
    infer.assert_awaited_once()


@pytest.mark.asyncio
async def test_whole_cohort_wire_correlates_each_reply_and_retires_receiver():
    receiving = asyncio.Event()
    retired = asyncio.Event()

    async def receive():
        receiving.set()
        try:
            await asyncio.Future()
        finally:
            retired.set()

    async def infer(cohort_id, members):
        await receiving.wait()
        return [{"actions": [1.0]}, None]

    async def send(_):
        assert retired.is_set()

    connection = object.__new__(RobotRealtimeConnection)
    connection.serving = SimpleNamespace(infer_whole_cohort=infer)
    connection.websocket = SimpleNamespace(receive=receive, send_bytes=AsyncMock(side_effect=send))
    assert await connection._infer_whole_cohort({"cohort_id": "c", "members": _members()})
    (reply,) = connection.websocket.send_bytes.await_args.args
    decoded = _unpack(reply)
    assert decoded == {
        "cohort_id": "c",
        "results": [
            {"request_id": "r-a", "payload": _pack({"actions": [1.0]})},
            {"request_id": "r-b", "error": "inference failed"},
        ],
    }


@pytest.mark.asyncio
async def test_disconnect_cancels_inference_and_preserves_replay_protection(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def infer(requests):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def disconnect():
        await started.wait()
        return {"type": "websocket.disconnect"}

    serving = await _serving(monkeypatch, infer)
    connection = object.__new__(RobotRealtimeConnection)
    connection.serving = serving
    connection.websocket = SimpleNamespace(receive=disconnect, send_bytes=AsyncMock())
    assert not await connection._infer_whole_cohort({"cohort_id": "c", "members": _members()})
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    connection.websocket.send_bytes.assert_not_awaited()
    with pytest.raises(ValueError, match="replayed"):
        await serving.infer_whole_cohort("c", _members())
