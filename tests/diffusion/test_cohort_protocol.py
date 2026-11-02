# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Neutral cohort scheduler protocol and close arithmetic."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from vllm_omni.diffusion.cohort_protocol import (
    COHORT_RECEIPT_FIELD,
    COHORT_WIRE_FIELD,
    COHORT_WIRE_SCHEMA,
    CohortProtocolError,
    CohortRegistry,
    cohort_forward_telemetry,
    cohort_receipt,
    evaluate_cohort_close,
    missing_cohort_receipt,
    parse_cohort_metadata,
)

POLICY_ID = "admission-policy-v1"
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _membership_hash(members: list[str]) -> str:
    canonical = json.dumps(sorted(members), ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _observation(size: int, member: str = "c0", *, sequence: int = 3) -> dict:
    members = [f"c{i}" for i in range(size)]
    request_id = f"request-{member}"
    return {
        "request_id": request_id,
        COHORT_WIRE_FIELD: {
            "schema": COHORT_WIRE_SCHEMA,
            "epoch_id": f"route/opaque/{sequence}",
            "member_ids": members,
            "membership_hash": _membership_hash(members),
            "deadline_ns": 900_000_000,
            "policy_id": POLICY_ID,
            "member_id": member,
            "request_id": request_id,
        },
    }


def _parse(observation: dict):
    return parse_cohort_metadata(
        observation,
        enabled=True,
        expected_policy_id=POLICY_ID,
        max_batch_size=4,
    )


@pytest.mark.parametrize("size", (1, 2, 3, 4))
def test_every_declared_batch_launches_at_final_member(size: int):
    registry = CohortRegistry()
    arrivals = {}
    for index in range(size):
        member = f"c{index}"
        metadata = _parse(_observation(size, member))
        assert metadata is not None
        registry.register(metadata)
        arrivals[member] = 100_000_000 + index * 10_000_000
        close = evaluate_cohort_close(
            metadata,
            arrivals,
            now_ns=arrivals[member],
            configured_timeout_ns=500_000_000,
            max_batch_size=4,
        )
        assert close.reason == ("expected_count" if index + 1 == size else None)
        assert close.arrived_count == index + 1
        assert close.missing_member_ids == tuple(f"c{missing}" for missing in range(index + 1, size))


def test_missing_member_close_precedence_is_deterministic():
    metadata = _parse(_observation(3))
    assert metadata is not None
    arrivals = {"c0": 100_000_000, "c1": 120_000_000}
    before = evaluate_cohort_close(
        metadata,
        arrivals,
        now_ns=899_999_999,
        configured_timeout_ns=1_000_000_000,
        max_batch_size=4,
    )
    at_deadline = evaluate_cohort_close(
        metadata,
        arrivals,
        now_ns=900_000_000,
        configured_timeout_ns=1_000_000_000,
        max_batch_size=4,
    )
    timeout_precedes_deadline = evaluate_cohort_close(
        metadata,
        arrivals,
        now_ns=200_000_000,
        configured_timeout_ns=200_000_000,
        max_batch_size=4,
    )
    capacity_metadata = _parse(_observation(4))
    assert capacity_metadata is not None
    at_capacity = evaluate_cohort_close(
        capacity_metadata,
        arrivals,
        now_ns=120_000_000,
        configured_timeout_ns=1_000_000_000,
        max_batch_size=2,
    )
    assert before.reason is None
    assert at_deadline.reason == "cohort_deadline"
    assert at_deadline.missing_member_ids == ("c2",)
    assert timeout_precedes_deadline.reason == "configured_timeout"
    assert at_capacity.reason == "max_batch"


@pytest.mark.parametrize(
    "field,value,reason",
    (
        ("schema", "vllm_omni.cohort.v0", "invalid_schema"),
        ("membership_hash", "0" * 64, "membership_hash_mismatch"),
        ("policy_id", "other", "policy_id_mismatch"),
        ("member_ids", ["c0", "c0"], "duplicate_members"),
        ("member_ids", ["c1", "c0"], "unsorted_members"),
    ),
)
def test_invalid_or_forged_metadata_is_rejected(field, value, reason):
    observation = _observation(3)
    observation[COHORT_WIRE_FIELD][field] = value
    with pytest.raises(CohortProtocolError, match=reason):
        _parse(observation)


def test_request_identity_must_match_transport_identity():
    observation = _observation(2)
    observation["request_id"] = "different"
    with pytest.raises(CohortProtocolError, match="request_id_mismatch"):
        _parse(observation)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (lambda observation: observation.__setitem__(COHORT_WIRE_FIELD, "invalid"), "invalid_metadata"),
        (
            lambda observation: observation[COHORT_WIRE_FIELD].__setitem__("member_id", "absent"),
            "member_not_in_cohort",
        ),
        (
            lambda observation: observation[COHORT_WIRE_FIELD].__setitem__(
                "member_ids", ["c0", "c1", "c2", "c3", "c4"]
            ),
            "invalid_expected_size",
        ),
        (
            lambda observation: observation[COHORT_WIRE_FIELD].__setitem__("deadline_ns", 0),
            "invalid_deadline_ns",
        ),
    ),
)
def test_malformed_metadata_is_rejected(mutate, reason):
    observation = _observation(3)
    mutate(observation)
    if reason == "invalid_expected_size":
        observation[COHORT_WIRE_FIELD]["membership_hash"] = _membership_hash(
            observation[COHORT_WIRE_FIELD]["member_ids"]
        )
    with pytest.raises(CohortProtocolError, match=reason):
        _parse(observation)


def test_partial_disabled_and_absent_metadata_are_distinct():
    observation = _observation(2)
    assert (
        parse_cohort_metadata(
            observation,
            enabled=False,
            expected_policy_id=None,
            max_batch_size=4,
        )
        is None
    )
    assert (
        parse_cohort_metadata(
            {},
            enabled=True,
            expected_policy_id=POLICY_ID,
            max_batch_size=4,
        )
        is None
    )
    with pytest.raises(CohortProtocolError, match="incomplete_metadata"):
        _parse({"request_id": "r", COHORT_WIRE_FIELD: {"schema": COHORT_WIRE_SCHEMA}})


def test_duplicate_conflict_interleaving_and_closed_epoch_are_rejected():
    registry = CohortRegistry()
    first = _parse(_observation(2, "c0"))
    second = _parse(_observation(2, "c1"))
    assert first is not None and second is not None
    registry.register(first)
    with pytest.raises(CohortProtocolError, match="duplicate_member"):
        registry.register(first)
    changed = _observation(2, "c1")
    changed[COHORT_WIRE_FIELD]["deadline_ns"] += 1
    conflicting = _parse(changed)
    assert conflicting is not None
    with pytest.raises(CohortProtocolError, match="conflicting_metadata"):
        registry.register(conflicting)
    registry.mark_uncoordinated()
    with pytest.raises(CohortProtocolError, match="noncontiguous_epoch"):
        registry.register(second)
    registry.close(first.epoch_id)
    with pytest.raises(CohortProtocolError, match="closed_epoch"):
        registry.register(second)


def test_failed_admission_rolls_back_registration():
    registry = CohortRegistry()
    first = _parse(_observation(2, "c0"))
    assert first is not None
    previous = registry.register(first)
    registry.rollback(first, previous)
    registry.register(first)


def test_forward_event_reports_exact_cohort_or_mixed_reason():
    metadata = _parse(_observation(2))
    assert metadata is not None
    close = evaluate_cohort_close(
        metadata,
        {"c0": 100, "c1": 110},
        now_ns=111,
        configured_timeout_ns=200,
        max_batch_size=4,
    )
    receipt = cohort_receipt(
        close,
        configured_batch_wait_ms=25,
        executed_count=2,
        forward_start_ns=112,
    )
    row = {COHORT_RECEIPT_FIELD: receipt}
    other = deepcopy(row)
    report = cohort_forward_telemetry([row, other])
    assert report["cohort_epoch_id"] == "route/opaque/3"
    assert report["cohort_executed_count"] == 2
    assert report["cohort_close_reason"] == "expected_count"
    assert report["cohort_forward_start_ns"] == 112
    other[COHORT_RECEIPT_FIELD]["epoch_id"] = "route/opaque/4"
    assert cohort_forward_telemetry([row, other]) == {"cohort_metadata_validation_reason": "mixed_epoch_or_membership"}
    assert cohort_forward_telemetry([{}]) == {}
    assert cohort_forward_telemetry([{COHORT_RECEIPT_FIELD: missing_cohort_receipt()}]) == {
        "cohort_metadata_validation_reason": "metadata_absent"
    }
