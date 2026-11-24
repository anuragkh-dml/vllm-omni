# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, fields
from typing import TYPE_CHECKING

from vllm_omni.diffusion.cohort_protocol import (
    COHORT_ELAPSED_SCHEMA,
    COHORT_RECEIPT_FIELD,
    COHORT_WIRE_SCHEMA,
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
_MAX_WHOLE_PROGRESS_REQUESTS = 4096

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
        schema = os.environ.get("VLLM_OMNI_COHORT_SCHEMA", COHORT_WIRE_SCHEMA)
        if schema not in (COHORT_WIRE_SCHEMA, COHORT_ELAPSED_SCHEMA) or (
            schema == COHORT_ELAPSED_SCHEMA and not enabled
        ):
            raise ValueError("relative cohort schema requires cohort scheduling")
        super().__init__()
        self._cohort_schema = schema
        self._cohort_enabled = enabled
        self._cohort_policy_id = policy_id
        self._cohorts = CohortRegistry()
        self._request_cohort: dict[str, CohortMetadata] = {}
        self._cohort_queue_enter_ns: dict[str, int] = {}
        self._cohort_configured_timeout_ns: int | None = None
        self._cohort_pending_close: CohortClose | None = None
        self._whole_cohorts: dict[str, tuple[str, ...]] = {}
        self._whole_progress: dict[str, dict[str, object]] = {}
        self._whole_cohort_members: dict[str, tuple[str, ...]] = {}
        self._whole_cohort_scopes: dict[str, str | None] = {}
        self._whole_request_scopes: dict[str, str | None] = {}
        self._whole_completed_request_ids: set[str] = set()
        self._whole_request_scheduled_ns: dict[str, int] = {}
        self._whole_progress_sequence = 1 if os.environ.get("VLLM_OMNI_OPENPI_PROGRESS") == "1" else 0
        self._whole_progress_lock = threading.Lock()
        self._whole_completion_events: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._whole_progress_enabled = os.environ.get("VLLM_OMNI_OPENPI_PROGRESS") == "1"
        self._reply_only_owner: str | None = None
        self._reply_only_cohort: str | None = None
        self._reply_only_fenced = False
        self._reply_only_required = os.environ.get("VLLM_OMNI_OPENPI_REPLY_ONLY_EXCLUSIVE") == "1"
        self._whole_progress_salt = uuid.uuid4().bytes if self._whole_progress_enabled else b""

    def _reset_scheduler_state(self) -> None:
        super()._reset_scheduler_state()
        self._cohorts = CohortRegistry()
        self._request_cohort.clear()
        self._cohort_queue_enter_ns.clear()
        self._cohort_configured_timeout_ns = None
        self._cohort_pending_close = None
        self._whole_cohorts.clear()
        with self._whole_progress_lock:
            self._whole_progress.clear()
            self._whole_cohort_members.clear()
            self._whole_cohort_scopes.clear()
            self._whole_request_scopes.clear()
            self._whole_completed_request_ids.clear()
            self._whole_request_scheduled_ns.clear()
            self._whole_completion_events.clear()
        # A normal OpenPI reset is not a process restart. Preserve the dedicated
        # owner fence, and fence any lease whose in-flight request was reset.
        if self._reply_only_required and self._reply_only_cohort is not None:
            self._reply_only_fenced = True

    def _pop_extra_request_state(self, request_id: str) -> None:
        super()._pop_extra_request_state(request_id)
        self._request_cohort.pop(request_id, None)
        self._cohort_queue_enter_ns.pop(request_id, None)
        self._whole_cohorts.pop(request_id, None)
        self._whole_request_scopes.pop(request_id, None)
        self._whole_completed_request_ids.discard(request_id)
        self._whole_request_scheduled_ns.pop(request_id, None)

    def add_whole_cohort(
        self,
        requests: list[OmniDiffusionRequest],
        *,
        cohort_id: str | None = None,
        reply_only_owner: str | None = None,
        progress_scope: str | None = None,
    ) -> tuple[str, ...]:
        """Validate every member before making the ready group visible to scheduling."""
        ids = tuple(request.request_id for request in requests)
        observed_cohort_id = (requests[0].sampling_params.extra_args or {}).get("whole_cohort_id") if requests else None
        cohort_id = cohort_id or observed_cohort_id
        if cohort_id is None and requests:
            # Preserve the scheduler's original cohort-less Python API. Serving
            # clients provide an explicit wire cohort ID; local callers get a
            # bounded process-unique identity for scheduler ownership only.
            cohort_id = f"local:{uuid.uuid4().hex}"
        if (
            not 1 <= len(ids) <= self.max_num_running_reqs
            or len(set(ids)) != len(ids)
            or not isinstance(cohort_id, str)
            or not cohort_id
            or (self._whole_progress_enabled and cohort_id in self._whole_cohort_members)
            or any(rid in self._request_states or rid in self._kv_draining_requests for rid in ids)
            or self._diffusion_kv_manager is not None
        ):
            raise ValueError("invalid whole-cohort size, identity or scheduler mode")
        if self._whole_progress_enabled and (
            not isinstance(progress_scope, str)
            or not 1 <= len(progress_scope) <= 128
            or any(
                char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
                for char in progress_scope
            )
        ):
            raise ValueError("progress-enabled whole cohort requires a valid progress scope")
        if self._whole_progress_enabled and len(self._whole_cohort_members) >= 1024:
            active_ids = set(self._running) | set(self._waiting)
            stale_id = next(
                (
                    old_id
                    for old_id, old_members in self._whole_cohort_members.items()
                    if not active_ids.intersection(old_members)
                ),
                None,
            )
            if stale_id is None:
                raise ValueError("whole-cohort progress registry is full")
            self._whole_cohort_members.pop(stale_id, None)
            self._whole_cohort_scopes.pop(stale_id, None)
            self._whole_progress.pop(stale_id, None)
        states = [self._make_request_state(rid, req) for rid, req in zip(ids, requests, strict=True)]
        if any(state.sampling_params_key != states[0].sampling_params_key for state in states[1:]):
            raise ValueError("whole cohort has incompatible sampling parameters")
        if reply_only_owner is not None:
            if not isinstance(reply_only_owner, str) or not 1 <= len(reply_only_owner) <= 256:
                raise ValueError("invalid reply-only owner identity")
            if not self._reply_only_required:
                raise ValueError("reply-only owner supplied to a shared scheduler")
            if (
                self._reply_only_fenced
                or (self._reply_only_owner is not None and self._reply_only_owner != reply_only_owner)
                or self._reply_only_cohort is not None
                or self._running
                or self._waiting
                or self._kv_draining_requests
            ):
                raise ValueError("reply-only scheduler is occupied or fenced")
            self._reply_only_owner = reply_only_owner
            self._reply_only_cohort = cohort_id
        elif self._reply_only_owner is not None:
            raise ValueError("scheduler is leased to a reply-only owner")
        elif self._reply_only_required:
            raise ValueError("dedicated reply-only scheduler requires an owner lease")
        queued_at = time.perf_counter()
        for state in states:
            rid = state.request_id
            state.queued_at = queued_at
            self._request_states[rid] = state
            self._whole_cohorts[rid] = ids
        self._waiting.extend(ids)
        if self._whole_progress_enabled:
            self._whole_cohort_members[cohort_id] = ids
            self._whole_cohort_scopes[cohort_id] = progress_scope
            for rid in ids:
                self._whole_request_scopes[rid] = progress_scope
            self._record_whole_cohort_progress(cohort_id)
        self._publish_benchmark_progress()
        return ids

    def whole_cohort_progress(self, cohort_id: str, progress_scope: str) -> dict[str, object]:
        """Return a fresh snapshot for one cohort within its admission-group scope."""
        if not self._whole_progress_enabled:
            raise RuntimeError("whole-cohort progress is not enabled")
        with self._whole_progress_lock:
            if cohort_id not in self._whole_cohort_members:
                raise KeyError("unknown whole cohort")
            if self._whole_cohort_scopes.get(cohort_id) != progress_scope:
                raise ValueError("progress scope does not own this whole cohort")
        return self._build_whole_cohort_progress_snapshot(cohort_id, progress_scope)

    def whole_cohort_progress_snapshot(self, progress_scope: str | None = None) -> dict[str, object]:
        """Return replica activity, exposing exact IDs only within a supplied scope."""
        if not self._whole_progress_enabled:
            raise RuntimeError("whole-cohort progress is not enabled")
        return self._build_whole_cohort_progress_snapshot(None, progress_scope)

    def _build_whole_cohort_progress_snapshot(
        self, cohort_id: str | None, progress_scope: str | None
    ) -> dict[str, object]:
        # Callers hold the engine condition lock. Scheduler-selected work that
        # has not entered a forward is queue-head work, not active model work.
        running = tuple(self._running)
        waiting = tuple(self._waiting)
        if len(running) + len(waiting) > _MAX_WHOLE_PROGRESS_REQUESTS:
            raise ValueError("whole-cohort progress snapshot exceeds request bound")
        forward = getattr(self, "_benchmark_active_forward", None)
        forward_ids = set(forward.get("internal_request_ids", ())) if forward else set()
        completed_internal = self._whole_completed_request_ids
        active_internal = tuple(rid for rid in running if rid in forward_ids and rid not in completed_internal)
        scheduled_internal = tuple(rid for rid in running if rid not in forward_ids and rid not in completed_internal)
        queued_internal = (*scheduled_internal, *waiting)
        scoped_members = {
            rid
            for rid, scope in self._whole_request_scopes.items()
            if progress_scope is not None and scope == progress_scope
        }
        now_ns = time.monotonic_ns()
        active_ids = [
            self._benchmark_progress_id(rid) if rid in scoped_members else self._opaque_progress_id(rid)
            for rid in active_internal
        ]
        queued_ids = [
            self._benchmark_progress_id(rid) if rid in scoped_members else self._opaque_progress_id(rid)
            for rid in queued_internal
        ]
        if cohort_id is None:
            state = "running" if forward else "scheduled" if scheduled_internal else "queued" if waiting else "idle"
        else:
            target_members = set(self._whole_cohort_members.get(cohort_id, ()))
            if forward and target_members.intersection(forward_ids):
                state = "running"
            elif target_members.intersection(scheduled_internal):
                state = "scheduled"
            elif target_members.intersection(waiting):
                state = "queued"
            else:
                state = "finished"

        active_start_age_ns = max(0, now_ns - int(forward["start_ns"])) if forward else None
        active_groups = {self._whole_cohorts[rid] for rid in active_internal if rid in self._whole_cohorts}
        queued_groups = {self._whole_cohorts[rid] for rid in queued_internal if rid in self._whole_cohorts}
        external_active = [rid for rid in active_internal if rid not in scoped_members]
        external_queued = [rid for rid in queued_internal if rid not in scoped_members]
        completions = []
        for forward_id, event in self._whole_completion_events.items():
            completion_ids = event["request_ids"]
            is_scoped_completion = progress_scope is not None and event.get("progress_scope") == progress_scope
            completions.append(
                {
                    "forward_id": forward_id,
                    "request_ids": (
                        list(completion_ids)
                        if is_scoped_completion
                        else [self._opaque_progress_id(rid) for rid in completion_ids]
                    ),
                    "age_ns": max(0, now_ns - int(event["ended_ns"])),
                }
            )
        return {
            "cohort_id": cohort_id,
            "progress_scope": progress_scope,
            "state": state,
            "active_forward_id": forward.get("forward_id") if active_internal and forward else None,
            "active_ids": active_ids,
            "queued_ids": queued_ids,
            "active_start_age_ns": active_start_age_ns if active_internal else None,
            "active_cohort_count": len(active_groups),
            "queued_cohort_count": len(queued_groups),
            "external_active_count": len(external_active),
            "external_queued_count": len(external_queued),
            "external_active_ids": [self._opaque_progress_id(rid) for rid in external_active],
            "external_queued_ids": [self._opaque_progress_id(rid) for rid in external_queued],
            "admission_pending_ids": [],
            "recent_completions": completions[-32:],
            "sequence": self._whole_progress_sequence,
            "observed_ns": now_ns,
        }

    def reply_only_is_idle(self) -> bool:
        """Prove the dedicated scheduler has no competing or in-flight work."""
        return (
            self._reply_only_required
            and not self._reply_only_fenced
            and self._reply_only_cohort is None
            and not self._running
            and not self._waiting
            and not self._kv_draining_requests
            and getattr(self, "_benchmark_active_forward", None) is None
        )

    def complete_reply_only_cohort(self, cohort_id: str, owner_id: str, *, terminal_received: bool) -> None:
        """Release a dedicated scheduler only after its reply delivery is certain."""
        if self._reply_only_cohort != cohort_id or self._reply_only_owner != owner_id:
            if terminal_received:
                raise ValueError("reply-only lease identity mismatch")
            self.fence_reply_only(owner_id)
            return
        if terminal_received:
            self._reply_only_cohort = None
        else:
            self._reply_only_fenced = True

    def fence_reply_only(self, owner_id: str) -> None:
        """Fail closed after an ambiguous exclusive-owner exchange."""
        if not self._reply_only_required:
            raise ValueError("cannot fence a shared scheduler as reply-only")
        if self._reply_only_owner is None:
            self._reply_only_owner = owner_id
        self._reply_only_fenced = True

    def _record_whole_cohort_progress(self, changed_cohort_id: str) -> None:
        if not self._whole_progress_enabled:
            return
        # This function is called under the engine condition lock. Store only a
        # compact transition projection; each poll takes a fresh timestamp and
        # age projection without advancing the event sequence.
        snapshot = self._build_whole_cohort_progress_snapshot(
            changed_cohort_id, self._whole_cohort_scopes.get(changed_cohort_id)
        )
        comparable = {key: value for key, value in snapshot.items() if key not in {"observed_ns", "sequence"}}
        with self._whole_progress_lock:
            previous = self._whole_progress.get(changed_cohort_id)
            if previous is not None and all(previous.get(key) == value for key, value in comparable.items()):
                return
            self._whole_progress_sequence += 1
            self._whole_progress[changed_cohort_id] = {
                **comparable,
                "sequence": self._whole_progress_sequence,
                "observed_ns": time.monotonic_ns(),
            }

    def _opaque_progress_id(self, request_id: str) -> str:
        digest = hmac.new(self._whole_progress_salt, request_id.encode(), hashlib.sha256).hexdigest()
        return f"external:{digest[:24]}"

    def _publish_all_whole_cohort_progress(self) -> None:
        for cohort_id in tuple(self._whole_cohort_members):
            self._record_whole_cohort_progress(cohort_id)

    def benchmark_forward_start(self, output: DiffusionSchedulerOutput) -> None:
        # A request can be scheduled again after a completed forward in engines
        # with multiple execution waves. The new forward makes it live again.
        for request_id in output.scheduled_request_ids:
            self._whole_completed_request_ids.discard(request_id)
        super().benchmark_forward_start(output)
        self._publish_all_whole_cohort_progress()

    def benchmark_forward_end(self) -> None:
        forward = getattr(self, "_benchmark_active_forward", None)
        if forward is not None:
            request_ids = tuple(forward.get("internal_request_ids", ()))
            # Whole-cohort request-mode inference returns one terminal result per
            # forward. The engine drains its payloads immediately afterward, so
            # these members must not masquerade as queued work in that window.
            self._whole_completed_request_ids.update(rid for rid in request_ids if rid in self._whole_request_scopes)
            scopes = {self._whole_request_scopes.get(rid) for rid in request_ids}
            progress_scope = next(iter(scopes)) if len(scopes) == 1 else None
            self._whole_completion_events[str(forward["forward_id"])] = {
                "request_ids": tuple(forward.get("request_ids", ())),
                "internal_request_ids": request_ids,
                "progress_scope": progress_scope,
                "ended_ns": time.monotonic_ns(),
            }
            while len(self._whole_completion_events) > 64:
                self._whole_completion_events.popitem(last=False)
        super().benchmark_forward_end()
        self._publish_all_whole_cohort_progress()

    def finish_requests(self, request_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        ids = [request_ids] if isinstance(request_ids, str) else request_ids
        # A cancelled queued member invalidates the entire ready unit. Once a
        # forward starts, normal per-request terminal handling owns its outputs.
        expanded = dict.fromkeys(
            member
            for rid in ids
            for member in (self._whole_cohorts.get(rid, (rid,)) if rid in self._waiting else (rid,))
        )
        super().finish_requests(list(expanded), status)
        self._publish_all_whole_cohort_progress()

    def add_startup_warmup(self, request: OmniDiffusionRequest) -> str:
        """Admit the engine's own dummy request before serving begins."""
        if self._reply_only_required:
            if not request.is_dummy_run() or self.has_requests():
                raise ValueError("startup warmup requires a dummy request and an idle scheduler")
            # Exclusive serving rejects every ordinary entry point. This one
            # startup-only dummy bypass is safe only before any work exists.
            return super()._add_request_with_request_id(request.request_id, request)
        if self._cohort_schema != COHORT_ELAPSED_SCHEMA:
            return self.add_request(request)
        if not request.is_dummy_run() or self.has_requests():
            raise ValueError("startup warmup requires a dummy request and an idle scheduler")
        # Only the engine's startup path calls this. Serving requests still
        # require v2 metadata even if their ID resembles the dummy request.
        return super()._add_request_with_request_id(request.request_id, request)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        if self._reply_only_required:
            raise ValueError("dedicated reply-only scheduler rejects requests without an owner lease")
        extra = getattr(request.sampling_params, "extra_args", None) or {}
        observation = extra.get("robot_obs")
        try:
            metadata = parse_cohort_metadata(
                observation if isinstance(observation, dict) else None,
                enabled=self._cohort_enabled,
                expected_policy_id=self._cohort_policy_id,
                max_batch_size=self.max_num_running_reqs,
                wire_schema=self._cohort_schema,
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
        self._publish_all_whole_cohort_progress()
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
                    obs = self._benchmark_observation(self._request_states[request_id])
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
                    obs = self._benchmark_observation(self._request_states[request_id])
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
        scheduled_at_ns = time.monotonic_ns()
        for request_id in output.scheduled_request_ids:
            if request_id in self._whole_cohorts:
                self._whole_request_scheduled_ns[request_id] = scheduled_at_ns
        self._publish_all_whole_cohort_progress()
        return output

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not super()._can_schedule_waiting(state):
            return False
        cohort = self._whole_cohorts.get(state.request_id)
        if self._running and cohort != self._whole_cohorts.get(self._running[0]):
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
        candidate_prefix_len = manager.get_metadata(state.request_id).sequences[0].cached_prefix_len
        return candidate_prefix_len == current

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
        if (
            max_wait_ms <= 0.0
            or self.max_num_running_reqs <= 1
            or (self._waiting and self._waiting[0] in self._whole_cohorts)
        ):
            return _AdmissionWaitDecision(should_wait=False)
        if self.num_running_requests() > 0:
            return _AdmissionWaitDecision(should_wait=False)

        max_wait_s = max_wait_ms / 1000.0
        stable_window_s = min(0.3, max_wait_s / 2.0) if dp_concurrent else min(0.05, max_wait_s / 5.0)
        configured_deadline = now + max_wait_s
        head = self._head_cohort()
        if head is not None:
            self._cohort_configured_timeout_ns = round(configured_deadline * 1e9)
            metadata, arrivals = head
            deadline = min(configured_deadline, metadata.server_deadline_ns(min(arrivals.values())) / 1e9)
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
        if self._waiting and self._waiting[0] in self._whole_cohorts:
            return True
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
