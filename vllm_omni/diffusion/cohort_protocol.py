# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Versioned cohort scheduling protocol and close arithmetic."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

COHORT_WIRE_FIELD = "vllm_omni_cohort"
COHORT_WIRE_SCHEMA = "vllm_omni.cohort.v1"
COHORT_RECEIPT_FIELD = "vllm_omni_server_cohort"
COHORT_RECEIPT_SCHEMA = "vllm_omni.cohort_receipt.v1"
COHORT_WIRE_FIELDS = frozenset(
    {
        "schema",
        "epoch_id",
        "member_ids",
        "membership_hash",
        "deadline_ns",
        "policy_id",
        "member_id",
        "request_id",
    }
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class CohortProtocolError(ValueError):
    """Invalid cohort metadata or a conflicting scheduler registration."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def membership_hash(member_ids: tuple[str, ...] | list[str]) -> str:
    """Hash a canonical membership independently of arrival order."""

    members = tuple(member_ids)
    if not members or any(not isinstance(item, str) or not item for item in members):
        raise CohortProtocolError("invalid_members")
    if len(set(members)) != len(members):
        raise CohortProtocolError("duplicate_members")
    canonical = json.dumps(sorted(members), ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise CohortProtocolError(f"invalid_{field}")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise CohortProtocolError(f"invalid_{field}")
    return value


def _integer(value: object, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CohortProtocolError(f"invalid_{field}")
    return value


@dataclass(frozen=True)
class CohortMetadata:
    epoch_id: str
    member_id: str
    request_id: str
    expected_size: int
    member_ids: tuple[str, ...]
    membership_hash: str
    deadline_ns: int
    policy_id: str

    @property
    def contract(self) -> tuple[object, ...]:
        return (
            self.epoch_id,
            self.expected_size,
            self.member_ids,
            self.membership_hash,
            self.deadline_ns,
            self.policy_id,
        )


def parse_cohort_metadata(
    observation: Mapping[str, object] | None,
    *,
    enabled: bool,
    expected_policy_id: str | None,
    max_batch_size: int,
) -> CohortMetadata | None:
    """Validate one request from the neutral, versioned wire protocol."""

    if not enabled or observation is None:
        return None
    if COHORT_WIRE_FIELD not in observation:
        return None
    wire = observation[COHORT_WIRE_FIELD]
    if not isinstance(wire, Mapping):
        raise CohortProtocolError("invalid_metadata")
    if not COHORT_WIRE_FIELDS <= wire.keys():
        raise CohortProtocolError("incomplete_metadata")
    if wire["schema"] != COHORT_WIRE_SCHEMA:
        raise CohortProtocolError("invalid_schema")
    expected_policy_id = _identifier(expected_policy_id, "configured_policy_id")
    epoch_id = _identifier(wire["epoch_id"], "epoch_id")
    member_id = _identifier(wire["member_id"], "member_id")
    request_id = _identifier(wire["request_id"], "request_id")
    if request_id != _identifier(observation.get("request_id"), "transport_request_id"):
        raise CohortProtocolError("request_id_mismatch")
    max_batch_size = _integer(max_batch_size, "max_batch_size", 1)
    raw_members = wire["member_ids"]
    if not isinstance(raw_members, (list, tuple)):
        raise CohortProtocolError("invalid_members")
    members = tuple(raw_members)
    digest = membership_hash(members)
    if members != tuple(sorted(members)):
        raise CohortProtocolError("unsorted_members")
    if len(members) > max_batch_size:
        raise CohortProtocolError("invalid_expected_size")
    if member_id not in members:
        raise CohortProtocolError("member_not_in_cohort")
    if _hash(wire["membership_hash"], "membership_hash") != digest:
        raise CohortProtocolError("membership_hash_mismatch")
    deadline_ns = _integer(wire["deadline_ns"], "deadline_ns", 1)
    policy_id = _identifier(wire["policy_id"], "policy_id")
    if policy_id != expected_policy_id:
        raise CohortProtocolError("policy_id_mismatch")
    return CohortMetadata(
        epoch_id=epoch_id,
        member_id=member_id,
        request_id=request_id,
        expected_size=len(members),
        member_ids=members,
        membership_hash=digest,
        deadline_ns=deadline_ns,
        policy_id=policy_id,
    )


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
    metadata: CohortMetadata,
    arrivals: Mapping[str, int],
    *,
    now_ns: int,
    configured_timeout_ns: int,
    max_batch_size: int,
) -> CohortClose:
    """Close at complete membership, capacity, action deadline, or timeout."""

    arrived = set(arrivals)
    if not arrived or not arrived <= set(metadata.member_ids):
        raise CohortProtocolError("invalid_arrivals")
    times = tuple(arrivals.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in times):
        raise CohortProtocolError("invalid_queue_enter_ns")
    now_ns = _integer(now_ns, "now_ns", 0)
    configured_timeout_ns = _integer(configured_timeout_ns, "configured_timeout_ns", 0)
    max_batch_size = _integer(max_batch_size, "max_batch_size", 1)
    reason = (
        "expected_count"
        if len(arrived) == metadata.expected_size
        else "max_batch"
        if len(arrived) >= max_batch_size
        else "cohort_deadline"
        if now_ns >= metadata.deadline_ns and metadata.deadline_ns <= configured_timeout_ns
        else "configured_timeout"
        if now_ns >= configured_timeout_ns
        else None
    )
    return CohortClose(
        metadata.epoch_id,
        metadata.membership_hash,
        metadata.expected_size,
        len(arrived),
        tuple(member for member in metadata.member_ids if member not in arrived),
        min(times),
        max(times),
        now_ns,
        reason,
    )


class CohortRegistry:
    """Reject duplicate, conflicting, and noncontiguous registrations."""

    def __init__(self) -> None:
        self._contracts: dict[str, tuple[object, ...]] = {}
        self._members: dict[str, set[str]] = {}
        self._closed: set[str] = set()
        self._last_epoch: str | None = None

    def register(self, metadata: CohortMetadata) -> str | None:
        epoch = metadata.epoch_id
        if epoch in self._closed:
            raise CohortProtocolError("closed_epoch")
        contract = self._contracts.get(epoch)
        if contract is not None and contract != metadata.contract:
            raise CohortProtocolError("conflicting_metadata")
        if contract is not None and self._last_epoch != epoch:
            raise CohortProtocolError("noncontiguous_epoch")
        members = self._members.setdefault(epoch, set())
        if metadata.member_id in members:
            raise CohortProtocolError("duplicate_member")
        previous_last = self._last_epoch
        self._contracts.setdefault(epoch, metadata.contract)
        members.add(metadata.member_id)
        self._last_epoch = epoch
        return previous_last

    def rollback(self, metadata: CohortMetadata, previous_last_epoch: str | None) -> None:
        """Undo registration when underlying request admission fails."""

        epoch = metadata.epoch_id
        members = self._members.get(epoch)
        if members is not None:
            members.discard(metadata.member_id)
            if not members:
                self._members.pop(epoch, None)
                self._contracts.pop(epoch, None)
        self._last_epoch = previous_last_epoch

    def mark_uncoordinated(self) -> None:
        self._last_epoch = None

    def close(self, epoch_id: str) -> None:
        if epoch_id in self._contracts:
            self._closed.add(epoch_id)


def cohort_receipt(
    close: CohortClose,
    *,
    configured_batch_wait_ms: float,
    executed_count: int | None = None,
    forward_start_ns: int | None = None,
) -> dict[str, object]:
    """Build a versioned server receipt for one closed cohort."""

    receipt: dict[str, object] = {
        "schema": COHORT_RECEIPT_SCHEMA,
        "epoch_id": close.epoch_id,
        "membership_hash": close.membership_hash,
        "expected_count": close.expected_count,
        "arrived_count": close.arrived_count,
        "missing_member_ids": list(close.missing_member_ids),
        "first_queue_enter_ns": close.first_queue_enter_ns,
        "last_queue_enter_ns": close.last_queue_enter_ns,
        "close_ns": close.close_ns,
        "close_reason": close.reason,
        "configured_batch_wait_ms": configured_batch_wait_ms,
        "actual_batch_wait_ms": (close.close_ns - close.first_queue_enter_ns) / 1e6,
        "metadata_validation_reason": "valid",
    }
    if executed_count is not None:
        receipt["executed_count"] = executed_count
    if forward_start_ns is not None:
        receipt["forward_start_ns"] = forward_start_ns
    return receipt


def missing_cohort_receipt() -> dict[str, object]:
    """Record that cohort scheduling was enabled but a request was uncoordinated."""

    return {
        "schema": COHORT_RECEIPT_SCHEMA,
        "metadata_validation_reason": "metadata_absent",
    }


def cohort_forward_telemetry(
    observations: list[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize scheduler receipts for one model forward telemetry event."""

    if not observations:
        return {}
    receipts = [observation.get(COHORT_RECEIPT_FIELD) for observation in observations]
    valid = [receipt for receipt in receipts if isinstance(receipt, Mapping) and receipt.get("epoch_id")]
    if not valid:
        reasons = {receipt.get("metadata_validation_reason") for receipt in receipts if isinstance(receipt, Mapping)}
        reasons.discard(None)
        if reasons:
            return {
                "cohort_metadata_validation_reason": (next(iter(reasons)) if len(reasons) == 1 else "mixed_reasons")
            }
        return {}
    first = valid[0]
    if len(valid) != len(observations) or any(
        receipt.get("epoch_id") != first.get("epoch_id")
        or receipt.get("membership_hash") != first.get("membership_hash")
        for receipt in valid
    ):
        return {"cohort_metadata_validation_reason": "mixed_epoch_or_membership"}
    fields = (
        "epoch_id",
        "membership_hash",
        "expected_count",
        "arrived_count",
        "missing_member_ids",
        "first_queue_enter_ns",
        "last_queue_enter_ns",
        "close_ns",
        "close_reason",
        "configured_batch_wait_ms",
        "actual_batch_wait_ms",
        "metadata_validation_reason",
        "forward_start_ns",
    )
    report = {f"cohort_{field}": first.get(field) for field in fields}
    report["cohort_executed_count"] = first.get("executed_count", len(observations))
    return report
