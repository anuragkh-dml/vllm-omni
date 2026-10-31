"""Benchmark-only expected-cohort wire contract and pure close arithmetic.

The server installs the same source as ``benchmark_expected_cohort.py``. A
missing or disabled contract always follows the ordinary scheduler path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Mapping, Any


COHORT_SCHEMA = "benchmark_expected_cohort.v1"
COHORT_FIELDS = frozenset({
    "benchmark_admission_epoch_id", "benchmark_expected_cohort_size",
    "benchmark_expected_member_ids", "benchmark_expected_membership_hash",
    "benchmark_cohort_deadline_ns", "benchmark_admission_policy_hash",
    "benchmark_run_id", "benchmark_server_id", "benchmark_client_id",
    "benchmark_request_id",
})
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class CohortViolation(ValueError):
    """Invalid cohort metadata or a conflicting scheduler registration."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def membership_hash(member_ids: tuple[str, ...] | list[str]) -> str:
    """Hash a canonical, sorted membership list (independent of arrival order)."""
    members = tuple(member_ids)
    if not members or any(not isinstance(item, str) or not item for item in members):
        raise CohortViolation("invalid_members")
    if len(set(members)) != len(members):
        raise CohortViolation("duplicate_members")
    canonical = json.dumps(sorted(members), ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CohortViolation(f"invalid_{field}")
    return value


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise CohortViolation(f"invalid_{field}")
    return value


def _integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CohortViolation(f"invalid_{field}")
    return value


def build_cohort_metadata(
    *, run_id: str, server_id: str, sequence: int,
    member_ids: tuple[str, ...] | list[str], client_id: str, request_id: str,
    deadline_ns: int, policy_hash: str, max_batch_size: int,
) -> dict[str, Any]:
    """Construct identical immutable cohort terms for every final member."""
    run_id = _identifier(run_id, "run_id")
    server_id = _identifier(server_id, "server_id")
    client_id = _identifier(client_id, "client_id")
    request_id = _identifier(request_id, "request_id")
    sequence = _integer(sequence, "sequence", 0)
    deadline_ns = _integer(deadline_ns, "deadline_ns", 1)
    max_batch_size = _integer(max_batch_size, "max_batch_size", 1)
    policy_hash = _hash(policy_hash, "policy_hash")
    digest = membership_hash(member_ids)
    members = tuple(sorted(member_ids))
    if client_id not in members:
        raise CohortViolation("client_not_member")
    if len(members) > max_batch_size:
        raise CohortViolation("cohort_exceeds_max_batch")
    return {
        "benchmark_admission_epoch_id": f"{run_id}:{server_id}:{sequence}",
        "benchmark_expected_cohort_size": len(members),
        "benchmark_expected_member_ids": list(members),
        "benchmark_expected_membership_hash": digest,
        "benchmark_cohort_deadline_ns": deadline_ns,
        "benchmark_admission_policy_hash": policy_hash,
        "benchmark_run_id": run_id,
        "benchmark_server_id": server_id,
        "benchmark_client_id": client_id,
        "benchmark_request_id": request_id,
    }


@dataclass(frozen=True)
class CohortMetadata:
    epoch_id: str
    run_id: str
    server_id: str
    client_id: str
    request_id: str
    expected_size: int
    member_ids: tuple[str, ...]
    membership_hash: str
    deadline_ns: int
    policy_hash: str

    @property
    def contract(self) -> tuple[Any, ...]:
        return (self.epoch_id, self.run_id, self.server_id,
                self.expected_size, self.member_ids, self.membership_hash,
                self.deadline_ns, self.policy_hash)


def parse_cohort_metadata(
    observation: Mapping[str, Any] | None, *, enabled: bool,
    expected_policy_hash: str | None, max_batch_size: int,
) -> CohortMetadata | None:
    """Validate an opted-in request; reject partial or forged contracts."""
    if not enabled or observation is None:
        return None
    present = COHORT_FIELDS & observation.keys()
    if not present:
        return None
    if present != COHORT_FIELDS:
        raise CohortViolation("incomplete_metadata")
    expected_policy_hash = _hash(expected_policy_hash, "configured_policy_hash")
    run_id = _identifier(observation["benchmark_run_id"], "run_id")
    server_id = _identifier(observation["benchmark_server_id"], "server_id")
    client_id = _identifier(observation["benchmark_client_id"], "client_id")
    request_id = _identifier(observation["benchmark_request_id"], "request_id")
    epoch_id = _identifier(observation["benchmark_admission_epoch_id"], "epoch_id")
    prefix = f"{run_id}:{server_id}:"
    if not epoch_id.startswith(prefix) or not epoch_id[len(prefix):].isdigit():
        raise CohortViolation("invalid_epoch_identity")
    max_batch_size = _integer(max_batch_size, "max_batch_size", 1)
    expected_size = _integer(observation["benchmark_expected_cohort_size"],
                             "expected_size", 1)
    raw_members = observation["benchmark_expected_member_ids"]
    if not isinstance(raw_members, (list, tuple)):
        raise CohortViolation("invalid_members")
    members = tuple(raw_members)
    digest = membership_hash(members)
    if members != tuple(sorted(members)):
        raise CohortViolation("unsorted_members")
    if expected_size != len(members) or expected_size > max_batch_size:
        raise CohortViolation("invalid_expected_size")
    if client_id not in members:
        raise CohortViolation("client_not_member")
    saved_digest = _hash(observation["benchmark_expected_membership_hash"],
                         "membership_hash")
    if saved_digest != digest:
        raise CohortViolation("membership_hash_mismatch")
    deadline_ns = _integer(observation["benchmark_cohort_deadline_ns"],
                           "deadline_ns", 1)
    policy_hash = _hash(observation["benchmark_admission_policy_hash"], "policy_hash")
    if policy_hash != expected_policy_hash:
        raise CohortViolation("policy_hash_mismatch")
    return CohortMetadata(epoch_id, run_id, server_id, client_id, request_id,
                          expected_size, members, digest, deadline_ns, policy_hash)


@dataclass(frozen=True)
class CohortClose:
    epoch_id: str
    membership_hash: str
    expected_count: int
    arrived_count: int
    missing_member_ids: tuple[str, ...]
    first_queue_enter_ns: int
    last_queue_enter_ns: int
    close_ns: int
    reason: str | None


def evaluate_cohort_close(
    metadata: CohortMetadata, arrivals: Mapping[str, int], *, now_ns: int,
    configured_timeout_ns: int,
    max_batch_size: int,
) -> CohortClose:
    """Close at complete membership, capacity, action deadline, or timeout."""
    arrived = set(arrivals)
    if not arrived or not arrived <= set(metadata.member_ids):
        raise CohortViolation("invalid_arrivals")
    times = tuple(arrivals.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in times):
        raise CohortViolation("invalid_queue_enter_ns")
    now_ns = _integer(now_ns, "now_ns", 0)
    configured_timeout_ns = _integer(configured_timeout_ns,
                                     "configured_timeout_ns", 0)
    max_batch_size = _integer(max_batch_size, "max_batch_size", 1)
    reason = ("expected_count" if len(arrived) == metadata.expected_size else
              "max_batch" if len(arrived) >= max_batch_size else
              "cohort_deadline" if now_ns >= metadata.deadline_ns
              and metadata.deadline_ns <= configured_timeout_ns else
              "configured_timeout" if now_ns >= configured_timeout_ns else None)
    return CohortClose(
        metadata.epoch_id, metadata.membership_hash,
        metadata.expected_size, len(arrived),
        tuple(member for member in metadata.member_ids if member not in arrived),
        min(times), max(times), now_ns, reason,
    )


class CohortRegistry:
    """Reject duplicate, conflicting, and noncontiguous registrations."""

    def __init__(self) -> None:
        self._contracts: dict[str, tuple[Any, ...]] = {}
        self._members: dict[str, set[str]] = {}
        self._closed: set[str] = set()
        self._last_epoch: str | None = None

    def register(self, metadata: CohortMetadata) -> str | None:
        epoch = metadata.epoch_id
        if epoch in self._closed:
            raise CohortViolation("closed_epoch")
        contract = self._contracts.get(epoch)
        if contract is not None and contract != metadata.contract:
            raise CohortViolation("conflicting_metadata")
        if contract is not None and self._last_epoch != epoch:
            raise CohortViolation("noncontiguous_epoch")
        members = self._members.setdefault(epoch, set())
        if metadata.client_id in members:
            raise CohortViolation("duplicate_member")
        previous_last = self._last_epoch
        self._contracts.setdefault(epoch, metadata.contract)
        members.add(metadata.client_id)
        self._last_epoch = epoch
        return previous_last

    def rollback(self, metadata: CohortMetadata, previous_last_epoch: str | None) -> None:
        """Undo registration when the underlying request admission fails."""
        epoch = metadata.epoch_id
        members = self._members.get(epoch)
        if members is not None:
            members.discard(metadata.client_id)
            if not members:
                self._members.pop(epoch, None)
                self._contracts.pop(epoch, None)
        self._last_epoch = previous_last_epoch

    def mark_legacy(self) -> None:
        self._last_epoch = None

    def close(self, epoch_id: str) -> None:
        if epoch_id in self._contracts:
            self._closed.add(epoch_id)


def forward_cohort_telemetry(observations: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize scheduler-stamped metadata for one model forward event."""
    if not observations:
        return {}
    if not any(
        observation.get("benchmark_server_cohort_epoch_id") for observation in observations
    ):
        reasons = {observation.get("benchmark_server_cohort_metadata_validation_reason")
                   for observation in observations}
        reasons.discard(None)
        if reasons:
            return {"cohort_metadata_validation_reason":
                    next(iter(reasons)) if len(reasons) == 1 else "mixed_legacy_reasons"}
        return {}
    fields = {
        "epoch_id": "benchmark_server_cohort_epoch_id",
        "membership_hash": "benchmark_server_cohort_membership_hash",
        "expected_count": "benchmark_server_cohort_expected_count",
        "arrived_count": "benchmark_server_cohort_arrived_count",
        "missing_member_ids": "benchmark_server_cohort_missing_member_ids",
        "first_queue_enter_ns": "benchmark_server_cohort_first_queue_enter_ns",
        "last_queue_enter_ns": "benchmark_server_cohort_last_queue_enter_ns",
        "close_ns": "benchmark_server_cohort_close_ns",
        "close_reason": "benchmark_server_cohort_close_reason",
        "configured_batch_wait_ms": "benchmark_server_cohort_configured_wait_ms",
        "actual_batch_wait_ms": "benchmark_server_cohort_actual_wait_ms",
        "metadata_validation_reason": "benchmark_server_cohort_metadata_validation_reason",
        "forward_start_ns": "benchmark_server_cohort_forward_start_ns",
    }
    first = observations[0]
    if not first.get("benchmark_server_cohort_epoch_id") or any(
        observation.get("benchmark_server_cohort_epoch_id") != first.get(
            "benchmark_server_cohort_epoch_id")
        or observation.get("benchmark_server_cohort_membership_hash") != first.get(
            "benchmark_server_cohort_membership_hash")
        for observation in observations
    ):
        return {"cohort_metadata_validation_reason": "mixed_epoch_or_membership"}
    report = {"cohort_" + name: first.get(source)
              for name, source in fields.items()}
    report["cohort_executed_count"] = len(observations)
    return report

