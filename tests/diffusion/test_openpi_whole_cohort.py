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

    async def infer(cohort_id, members, *, owner_id=None, progress=False):
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

    async def infer(requests, *, cohort_id=None, reply_only_owner=None):
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


def test_progress_snapshot_preserves_fifo_and_hides_external_ids(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_OPENPI_PROGRESS", "1")
    scheduler = _scheduler(monkeypatch)
    own = [_request("internal-a"), _request("internal-b")]
    own[0].sampling_params.extra_args["robot_obs"] = {"request_id": "robot-a"}
    own[1].sampling_params.extra_args["robot_obs"] = {"request_id": "robot-b"}
    scheduler.add_whole_cohort(own, cohort_id="cohort-a")
    scheduler.add_request(_request("secret-external-request"))

    snapshot = scheduler.whole_cohort_progress("cohort-a")
    assert snapshot["state"] == "queued"
    assert snapshot["queued_ids"][:2] == ["robot-a", "robot-b"]
    assert snapshot["queued_ids"][2].startswith("external:")
    assert "secret-external-request" not in repr(snapshot)
    assert snapshot["external_queued_ids"] == [snapshot["queued_ids"][2]]
    assert snapshot["external_queued_count"] == 1
    assert snapshot["queued_cohort_count"] == 1
    assert scheduler.whole_cohort_progress("cohort-a")["sequence"] == snapshot["sequence"]

    output = scheduler.schedule()
    scheduler.benchmark_forward_start(output)
    running = scheduler.whole_cohort_progress("cohort-a")
    assert running["state"] == "running"
    assert running["active_ids"] == ["robot-a", "robot-b"]
    assert running["active_forward_id"] is not None
    assert running["active_start_age_ns"] is not None
    assert running["external_queued_count"] == 1
    assert running["queued_ids"][0] == snapshot["queued_ids"][2]
    assert running["sequence"] > snapshot["sequence"]
    scheduler.benchmark_forward_end()
    completed = scheduler.whole_cohort_progress("cohort-a")["recent_completions"]
    assert completed[-1]["request_ids"] == ["robot-a", "robot-b"]


def test_reply_only_scheduler_rejects_shared_entry_and_fences_ambiguous_owner(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_OPENPI_REPLY_ONLY_EXCLUSIVE", "1")
    scheduler = _scheduler(monkeypatch)
    assert scheduler.reply_only_is_idle()
    with pytest.raises(ValueError, match="without an owner"):
        scheduler.add_request(_request("ordinary"))

    scheduler.add_whole_cohort([_request("owned")], cohort_id="cohort-a", reply_only_owner="owner-a")
    with pytest.raises(ValueError, match="occupied or fenced"):
        scheduler.add_whole_cohort([_request("other")], cohort_id="cohort-b", reply_only_owner="owner-b")
    with pytest.raises(ValueError, match="without an owner"):
        scheduler.add_request(_request("ordinary-again"))

    assert not scheduler.reply_only_is_idle()
    scheduler.complete_reply_only_cohort("cohort-a", "owner-a", terminal_received=False)
    scheduler._reset_scheduler_state()
    assert scheduler._reply_only_fenced
    with pytest.raises(ValueError, match="occupied or fenced"):
        scheduler.add_whole_cohort([_request("retry")], cohort_id="cohort-c", reply_only_owner="owner-a")


def test_progress_poll_returns_terminal_instead_of_late_progress():
    async def infer(cohort_id, members, *, owner_id=None, progress=False):
        await asyncio.sleep(0)
        return [{"actions": [1.0]}, {"actions": [2.0]}]

    calls = 0
    poll = {
        "type": "progress_poll",
        "schema": "vllm_omni.infer_cohort_progress.v1",
        "instance_id": "instance",
        "generation": "generation",
        "cohort_id": "c",
        "poll_sequence": 1,
    }

    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "websocket.receive", "bytes": _pack(poll)}
        raise AssertionError("server waited for a poll after terminal")

    connection = object.__new__(RobotRealtimeConnection)
    connection.serving = SimpleNamespace(
        _process_generation="generation",
        infer_whole_cohort=infer,
        whole_cohort_progress=Mock(),
        complete_reply_only_cohort=Mock(),
    )
    connection.websocket = SimpleNamespace(
        receive=receive,
        send_bytes=AsyncMock(),
    )
    import os

    old_instance = os.environ.get("VLLM_OMNI_STATUS_INSTANCE")
    os.environ["VLLM_OMNI_STATUS_INSTANCE"] = "instance"
    try:
        assert asyncio.run(connection._infer_whole_cohort({"cohort_id": "c", "members": _members(), "progress": True}))
    finally:
        if old_instance is None:
            os.environ.pop("VLLM_OMNI_STATUS_INSTANCE", None)
        else:
            os.environ["VLLM_OMNI_STATUS_INSTANCE"] = old_instance
    connection.serving.whole_cohort_progress.assert_not_called()
    assert calls == 1


@pytest.mark.asyncio
async def test_progress_exchange_has_one_frame_per_poll_and_terminal(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_STATUS_INSTANCE", "instance")
    inference_done = asyncio.Event()
    polls = [1, 2]

    async def infer(cohort_id, members, *, owner_id=None, progress=False):
        await inference_done.wait()
        return [{"actions": [1.0]}, {"actions": [2.0]}]

    async def receive():
        sequence = polls.pop(0)
        return {
            "type": "websocket.receive",
            "bytes": _pack(
                {
                    "type": "progress_poll",
                    "schema": "vllm_omni.infer_cohort_progress.v1",
                    "instance_id": "instance",
                    "generation": "generation",
                    "cohort_id": "c",
                    "poll_sequence": sequence,
                }
            ),
        }

    progress = Mock(
        side_effect=lambda cohort_id, sequence, pending_ids: (
            inference_done.set()
            or {
                "type": "progress",
                "cohort_id": cohort_id,
                "poll_sequence": sequence,
                "state": "running",
            }
        )
    )
    connection = object.__new__(RobotRealtimeConnection)
    connection.serving = SimpleNamespace(
        _process_generation="generation",
        infer_whole_cohort=infer,
        whole_cohort_progress=progress,
        complete_reply_only_cohort=Mock(),
    )
    connection.websocket = SimpleNamespace(receive=receive, send_bytes=AsyncMock())

    assert await connection._infer_whole_cohort({"cohort_id": "c", "members": _members(), "progress": True})
    frames = [_unpack(args.args[0]) for args in connection.websocket.send_bytes.await_args_list]
    assert [frame.get("type", "terminal") for frame in frames] == ["progress", "terminal"]
    assert frames[0]["poll_sequence"] == 1
    assert frames[1]["cohort_id"] == "c"
    progress.assert_called_once_with("c", 1, ["r-a", "r-b"])


def test_progress_uses_admitting_state_until_scheduler_enqueue(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_OPENPI_PROGRESS", "1")
    monkeypatch.setenv("VLLM_OMNI_STATUS_INSTANCE", "instance")
    snapshot = {
        "cohort_id": None,
        "state": "idle",
        "sequence": 3,
        "observed_ns": 100,
        "active_ids": [],
        "queued_ids": [],
        "external_active_count": 0,
        "external_queued_count": 0,
    }
    replica = SimpleNamespace(
        whole_cohort_capacity=4,
        whole_cohort_progress=Mock(side_effect=KeyError("not enqueued yet")),
        whole_cohort_progress_snapshot=Mock(return_value=dict(snapshot)),
    )
    serving = object.__new__(ServingRealtimeRobotOpenPI)
    serving._whole_cohort_replica = lambda: replica
    serving._process_generation = "generation"

    reply = serving.whole_cohort_progress("cohort-a", 7, ["request-a", "request-b"])
    assert reply["type"] == "progress"
    assert reply["state"] == "admitting"
    assert reply["cohort_id"] == "cohort-a"
    assert reply["poll_sequence"] == 7
    assert reply["active_ids"] == []
    assert reply["queued_ids"] == []
    assert reply["admission_pending_ids"] == ["request-a", "request-b"]
