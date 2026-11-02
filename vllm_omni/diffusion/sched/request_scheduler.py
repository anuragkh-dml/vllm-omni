# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import time
from dataclasses import asdict, fields
from typing import TYPE_CHECKING

from vllm_omni.diffusion.cohort_protocol import (
    COHORT_RECEIPT_FIELD,
    CohortClose,
    CohortMetadata,
    CohortProtocolError,
    CohortRegistry,
    cohort_receipt,
    evaluate_cohort_close,
    missing_cohort_receipt,
    parse_cohort_metadata,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    RequestBatchSamplingParamsKey,
    SchedulerRequestState,
    _AdmissionWaitDecision,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(
    field.name for field in fields(RequestBatchSamplingParamsKey)
) - {"condition_key", "flow_shift", "lora_int_id", "sample_solver"}


def _normalize_explicit_sample_solver(value: object | None) -> str | None:
    """Normalize an explicitly provided solver without selecting a default."""
    if value is None:
        return None
    return str(value).strip().lower()


def _normalize_explicit_flow_shift(value: float | str | None) -> float | None:
    """Normalize an explicitly provided flow shift without selecting a default."""
    if value is None:
        return None
    return float(value)


def build_request_batch_sampling_params_key(request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
    """Build the compatibility key shared by scheduling and DP dispatch."""
    sampling = request.sampling_params
    # LoRA identity is optional on sampling params (and on test stubs).
    lora_request = getattr(sampling, "lora_request", None)
    key_kwargs = {name: getattr(sampling, name) for name in _REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES}
    extra_args = sampling.extra_args or {}
    # Match pipeline resolution for explicit overrides, but preserve None:
    # pipeline/engine defaults are configuration-dependent and must not be
    # inferred while building the request-batch key.
    key_kwargs["sample_solver"] = _normalize_explicit_sample_solver(extra_args.get("sample_solver"))
    key_kwargs["flow_shift"] = _normalize_explicit_flow_shift(extra_args.get("flow_shift"))
    key_kwargs["condition_key"] = getattr(request, "batch_compatibility_key", None)
    key_kwargs["lora_int_id"] = lora_request.lora_int_id if lora_request is not None else None
    return RequestBatchSamplingParamsKey(**key_kwargs)


class RequestScheduler(BaseScheduler):
    """Scheduler for static request waves, including admission coalescing."""

    def __init__(self) -> None:
        enabled = os.environ.get("VLLM_OMNI_COHORT_SCHEDULING") == "1"
        policy_id = os.environ.get("VLLM_OMNI_COHORT_POLICY_ID")
        if enabled and (not isinstance(policy_id, str) or not policy_id or len(policy_id) > 256):
            raise ValueError("cohort scheduling requires a nonempty policy ID")
        super().__init__()
        self._cohort_enabled = enabled
        self._cohort_policy_id = policy_id
        self._cohorts = CohortRegistry()
        self._request_cohort: dict[str, CohortMetadata] = {}
        self._cohort_queue_enter_ns: dict[str, int] = {}
        self._cohort_configured_timeout_ns: int | None = None
        self._cohort_pending_close: CohortClose | None = None

    def _reset_scheduler_state(self) -> None:
        super()._reset_scheduler_state()
        self._cohorts = CohortRegistry()
        self._request_cohort.clear()
        self._cohort_queue_enter_ns.clear()
        self._cohort_configured_timeout_ns = None
        self._cohort_pending_close = None

    def _pop_extra_request_state(self, request_id: str) -> None:
        super()._pop_extra_request_state(request_id)
        self._request_cohort.pop(request_id, None)
        self._cohort_queue_enter_ns.pop(request_id, None)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        extra = getattr(request.sampling_params, "extra_args", None) or {}
        observation = extra.get("robot_obs")
        try:
            metadata = parse_cohort_metadata(
                observation if isinstance(observation, dict) else None,
                enabled=self._cohort_enabled,
                expected_policy_id=self._cohort_policy_id,
                max_batch_size=self.max_num_running_reqs,
            )
            previous_last = self._cohorts.register(metadata) if metadata is not None else None
        except CohortProtocolError as exc:
            feed = getattr(self, "_benchmark_progress", None)
            if feed is not None:
                feed.record_metadata_rejection(
                    {
                        "at_ns": time.monotonic_ns(),
                        "request_id": request_id,
                        "reason": exc.reason,
                    }
                )
            raise
        try:
            admitted = super()._add_request_with_request_id(request_id, request)
        except Exception:
            if metadata is not None:
                self._cohorts.rollback(metadata, previous_last)
            raise
        if metadata is not None:
            self._request_cohort[admitted] = metadata
            self._cohort_queue_enter_ns[admitted] = time.monotonic_ns()
        elif self._cohort_enabled:
            self._cohorts.mark_uncoordinated()
            if isinstance(observation, dict):
                observation[COHORT_RECEIPT_FIELD] = missing_cohort_receipt()
        return admitted

    def _head_cohort(self) -> tuple[CohortMetadata, dict[str, int]] | None:
        if not self._cohort_enabled or not self._waiting:
            return None
        first = self._request_cohort.get(self._waiting[0])
        if first is None:
            return None
        arrivals: dict[str, int] = {}
        for request_id in self._waiting:
            metadata = self._request_cohort.get(request_id)
            if metadata is None or metadata.contract != first.contract:
                break
            arrivals[metadata.member_id] = self._cohort_queue_enter_ns[request_id]
        return first, arrivals

    def _close_head_cohort(self, *, now_ns: int, configured_timeout_ns: int) -> CohortClose | None:
        head = self._head_cohort()
        if head is None:
            return None
        metadata, arrivals = head
        return evaluate_cohort_close(
            metadata,
            arrivals,
            now_ns=now_ns,
            configured_timeout_ns=configured_timeout_ns,
            max_batch_size=self.max_num_running_reqs,
        )

    def schedule(self) -> DiffusionSchedulerOutput:
        head = self._head_cohort()
        close: CohortClose | None = None
        member_request_ids: list[str] = []
        if head is not None:
            metadata, _ = head
            now_ns = time.monotonic_ns()
            close = self._cohort_pending_close
            if close is None or close.epoch_id != metadata.epoch_id:
                close = self._close_head_cohort(now_ns=now_ns, configured_timeout_ns=now_ns)
            if close is not None:
                reason = close.reason or "configured_timeout"
                close = CohortClose(
                    close.epoch_id,
                    close.membership_hash,
                    close.expected_count,
                    close.arrived_count,
                    close.missing_member_ids,
                    close.first_queue_enter_ns,
                    close.last_queue_enter_ns,
                    now_ns,
                    reason,
                )
                for request_id in self._waiting:
                    member = self._request_cohort.get(request_id)
                    if member is None or member.contract != metadata.contract:
                        break
                    member_request_ids.append(request_id)
                    obs = self._benchmark_trace_obs(self._request_states[request_id])
                    if obs is not None:
                        obs[COHORT_RECEIPT_FIELD] = cohort_receipt(
                            close,
                            configured_batch_wait_ms=(
                                self.od_config.request_batch_max_wait_ms if self.od_config else 0
                            ),
                        )
        output = super().schedule()
        if close is not None and member_request_ids:
            executed = set(output.scheduled_request_ids)
            actual = [request_id for request_id in member_request_ids if request_id in executed]
            if actual:
                for request_id in actual:
                    obs = self._benchmark_trace_obs(self._request_states[request_id])
                    if obs is not None:
                        obs[COHORT_RECEIPT_FIELD]["executed_count"] = len(actual)
                self._cohorts.close(close.epoch_id)
                feed = getattr(self, "_benchmark_progress", None)
                if feed is not None:
                    feed.record_cohort_close(
                        {
                            **asdict(close),
                            "executed_count": len(actual),
                            "request_ids": [self._request_cohort[rid].request_id for rid in actual],
                        }
                    )
        self._cohort_pending_close = None
        return output

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not super()._can_schedule_waiting(state):
            return False
        if self._cohort_enabled and self._running:
            first = self._request_cohort.get(self._running[0])
            candidate = self._request_cohort.get(state.request_id)
            if (first is None) != (candidate is None) or (
                first is not None and candidate is not None and first.contract != candidate.contract
            ):
                return False
        manager = self._diffusion_kv_manager
        if not self._running or manager is None or not manager.has_request(state.request_id):
            return True
        # The request-level runner uses one first-step query boundary per wave.
        # The Manager already aligns all CFG sequences within each request.
        current = manager.get_metadata(self._running[0]).sequences[0].cached_prefix_len
        candidate = manager.get_metadata(state.request_id).sequences[0].cached_prefix_len
        return candidate == current

    def get_admission_wait_decision(
        self,
        *,
        now: float,
        dp_concurrent: bool = False,
    ) -> _AdmissionWaitDecision:
        assert self.od_config is not None
        self._cohort_pending_close = None
        self._cohort_configured_timeout_ns = None
        max_wait_ms = self.od_config.request_batch_max_wait_ms
        if max_wait_ms <= 0.0 or self.max_num_running_reqs <= 1:
            return _AdmissionWaitDecision(should_wait=False)
        if self.num_running_requests() > 0:
            return _AdmissionWaitDecision(should_wait=False)

        max_wait_s = max_wait_ms / 1000.0
        stable_window_s = min(0.3, max_wait_s / 2.0) if dp_concurrent else min(0.05, max_wait_s / 5.0)
        configured_deadline = now + max_wait_s
        head = self._head_cohort()
        if head is not None:
            self._cohort_configured_timeout_ns = round(configured_deadline * 1e9)
            deadline = min(configured_deadline, head[0].deadline_ns / 1e9)
            return _AdmissionWaitDecision(
                should_wait=True,
                deadline=deadline,
                stable_window_s=0.0,
                max_batch=self.max_num_running_reqs,
            )
        return _AdmissionWaitDecision(
            should_wait=True,
            deadline=configured_deadline,
            stable_window_s=stable_window_s,
            max_batch=self.max_num_running_reqs,
        )

    def should_end_admission_wait(
        self,
        decision: _AdmissionWaitDecision,
        *,
        now: float,
        stable_since: float,
    ) -> bool:
        head = self._head_cohort()
        if head is not None:
            now_ns = round(now * 1e9)
            close = self._close_head_cohort(
                now_ns=now_ns,
                configured_timeout_ns=(
                    self._cohort_configured_timeout_ns if self._cohort_configured_timeout_ns is not None else now_ns
                ),
            )
            if close is not None and close.reason is not None:
                self._cohort_pending_close = close
                return True
            return False
        waiting = self.num_waiting_requests()
        return (
            waiting >= decision.max_batch
            or (waiting > 0 and now - stable_since >= decision.stable_window_s)
            or (decision.deadline is not None and now >= decision.deadline)
        )

    def _build_sampling_params_key(self, request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
        return build_request_batch_sampling_params_key(request)

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids and not sched_output.finished_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            req_output = output.get_request_output(request_id)
            result = req_output.result if req_output is not None else None
            if result is None:
                # Async mode: result=None with async_output_id means compute done,
                # final output will arrive later via wait_output_ready.
                if req_output is not None and req_output.async_output_id is not None:
                    terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                    terminal_errors[request_id] = None
                else:
                    terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                    terminal_errors[request_id] = "No output result"
            elif result.aborted:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ABORTED
                terminal_errors[request_id] = None
            elif result.error:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = result.error
            else:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
