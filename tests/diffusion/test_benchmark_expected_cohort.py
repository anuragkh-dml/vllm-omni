"""Expected-cohort scheduler contract and close arithmetic."""

from __future__ import annotations

from copy import deepcopy

import pytest

from vllm_omni.diffusion.benchmark_expected_cohort import (
    CohortRegistry,
    CohortViolation,
    build_cohort_metadata,
    evaluate_cohort_close,
    forward_cohort_telemetry,
    parse_cohort_metadata,
)

POLICY_HASH = "a" * 64


def _metadata(size: int, client: str = "c0", *, sequence: int = 3) -> dict:
    return build_cohort_metadata(
        run_id="run", server_id="route", sequence=sequence,
        member_ids=[f"c{i}" for i in range(size)], client_id=client,
        request_id=f"request-{client}", deadline_ns=900_000_000,
        policy_hash=POLICY_HASH, max_batch_size=4,
    )


def _parse(row: dict):
    return parse_cohort_metadata(
        row, enabled=True, expected_policy_hash=POLICY_HASH, max_batch_size=4,
    )


@pytest.mark.parametrize("size", (1, 2, 3, 4))
def test_every_declared_batch_launches_at_final_member(size: int):
    registry = CohortRegistry()
    arrivals = {}
    for index in range(size):
        client = f"c{index}"
        metadata = _parse(_metadata(size, client))
        assert metadata is not None
        registry.register(metadata)
        arrivals[client] = 100_000_000 + index * 10_000_000
        close = evaluate_cohort_close(
            metadata, arrivals, now_ns=arrivals[client],
            configured_timeout_ns=500_000_000, max_batch_size=4,
        )
        assert close.reason == ("expected_count" if index + 1 == size else None)
        assert close.arrived_count == index + 1
        assert close.missing_member_ids == tuple(
            f"c{missing}" for missing in range(index + 1, size)
        )


def test_missing_member_closes_at_deadline_or_configured_timeout():
    metadata = _parse(_metadata(3))
    assert metadata is not None
    arrivals = {"c0": 100_000_000, "c1": 120_000_000}
    before = evaluate_cohort_close(
        metadata, arrivals, now_ns=899_999_999,
        configured_timeout_ns=1_000_000_000, max_batch_size=4,
    )
    at_deadline = evaluate_cohort_close(
        metadata, arrivals, now_ns=900_000_000,
        configured_timeout_ns=1_000_000_000, max_batch_size=4,
    )
    at_timeout = evaluate_cohort_close(
        metadata, arrivals, now_ns=200_000_000,
        configured_timeout_ns=200_000_000, max_batch_size=4,
    )
    assert before.reason is None
    assert at_deadline.reason == "cohort_deadline"
    assert at_deadline.missing_member_ids == ("c2",)
    assert at_timeout.reason == "configured_timeout"


@pytest.mark.parametrize("field,value,reason", (
    ("benchmark_expected_cohort_size", 0, "invalid_expected_size"),
    ("benchmark_expected_cohort_size", 5, "invalid_expected_size"),
    ("benchmark_expected_cohort_size", 1, "invalid_expected_size"),
    ("benchmark_admission_epoch_id", "other:route:3", "invalid_epoch_identity"),
    ("benchmark_expected_membership_hash", "0" * 64, "membership_hash_mismatch"),
    ("benchmark_admission_policy_hash", "b" * 64, "policy_hash_mismatch"),
    ("benchmark_expected_member_ids", ["c0", "c0"], "duplicate_members"),
))
def test_invalid_or_forged_metadata_rejected(field, value, reason):
    row = _metadata(3)
    row[field] = value
    with pytest.raises(CohortViolation, match=reason):
        _parse(row)


def test_partial_and_disabled_metadata_keep_legacy_behavior_distinct():
    row = _metadata(2)
    assert parse_cohort_metadata(
        row, enabled=False, expected_policy_hash=None, max_batch_size=4,
    ) is None
    assert parse_cohort_metadata(
        {}, enabled=True, expected_policy_hash=POLICY_HASH, max_batch_size=4,
    ) is None
    with pytest.raises(CohortViolation, match="incomplete_metadata"):
        _parse({"benchmark_expected_cohort_size": 2})


def test_duplicate_conflict_interleaving_and_closed_epoch_are_rejected():
    registry = CohortRegistry()
    first = _parse(_metadata(2, "c0"))
    second = _parse(_metadata(2, "c1"))
    assert first is not None and second is not None
    registry.register(first)
    with pytest.raises(CohortViolation, match="duplicate_member"):
        registry.register(first)
    changed = _metadata(2, "c1")
    changed["benchmark_cohort_deadline_ns"] += 1
    conflicting = _parse(changed)
    assert conflicting is not None
    with pytest.raises(CohortViolation, match="conflicting_metadata"):
        registry.register(conflicting)
    registry.mark_legacy()
    with pytest.raises(CohortViolation, match="noncontiguous_epoch"):
        registry.register(second)
    registry.close(first.epoch_id)
    with pytest.raises(CohortViolation, match="closed_epoch"):
        registry.register(second)


def test_failed_admission_rolls_back_registration():
    registry = CohortRegistry()
    first = _parse(_metadata(2, "c0"))
    assert first is not None
    previous = registry.register(first)
    registry.rollback(first, previous)
    registry.register(first)


def test_forward_event_reports_exact_cohort_or_mixed_reason():
    row = _metadata(2)
    row.update(
        benchmark_server_cohort_epoch_id=row["benchmark_admission_epoch_id"],
        benchmark_server_cohort_membership_hash=row[
            "benchmark_expected_membership_hash"
        ],
        benchmark_server_cohort_expected_count=2,
        benchmark_server_cohort_arrived_count=2,
        benchmark_server_cohort_missing_member_ids=[],
        benchmark_server_cohort_first_queue_enter_ns=100,
        benchmark_server_cohort_last_queue_enter_ns=110,
        benchmark_server_cohort_close_ns=111,
        benchmark_server_cohort_close_reason="expected_count",
        benchmark_server_cohort_configured_wait_ms=25,
        benchmark_server_cohort_actual_wait_ms=0.000011,
        benchmark_server_cohort_metadata_validation_reason="valid",
        benchmark_server_cohort_forward_start_ns=112,
    )
    other = deepcopy(row)
    other["benchmark_client_id"] = "c1"
    report = forward_cohort_telemetry([row, other])
    assert report["cohort_epoch_id"] == "run:route:3"
    assert report["cohort_executed_count"] == 2
    assert report["cohort_close_reason"] == "expected_count"
    assert report["cohort_forward_start_ns"] == 112
    other["benchmark_server_cohort_epoch_id"] = "run:route:4"
    assert forward_cohort_telemetry([row, other]) == {
        "cohort_metadata_validation_reason": "mixed_epoch_or_membership"
    }
    assert forward_cohort_telemetry([{}]) == {}
    assert forward_cohort_telemetry([{
        "benchmark_server_cohort_metadata_validation_reason": "metadata_absent"
    }]) == {"cohort_metadata_validation_reason": "metadata_absent"}
