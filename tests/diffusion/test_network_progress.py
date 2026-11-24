# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import http.client
import json
import secrets

import pytest

from vllm_omni.diffusion.network_progress import ABSOLUTE, ELAPSED, NetworkProgressEndpoint, _mac

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _poll(port, key, schema=ABSOLUTE):
    nonce = secrets.token_hex(32).encode()
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        client.request(
            "GET",
            "/status",
            headers={
                "X-Status-Nonce": nonce.decode(),
                "X-Status-Schema": schema,
                "X-Status-Auth": _mac(key, b"request", nonce, schema.encode()).decode(),
            },
        )
        reply = client.getresponse()
        body = reply.read()
        if reply.status == 200:
            assert reply.getheader("X-Status-Auth") == _mac(key, b"response", nonce, schema.encode(), body).decode()
        return reply.status, body
    finally:
        client.close()


def _snapshot(**fields):
    return {
        "schema": "omni_scheduler_progress.v1",
        "clock_domain": "host_monotonic_ns",
        "hostname": "policy-host",
        "observed_ns": 10**15,
        "active": None,
        "queued_request_ids": [],
        "recent_events": [],
        **fields,
    }


def test_status_authentication_order_and_server_restart():
    key = bytes(range(32))
    first = NetworkProgressEndpoint(0, key.hex(), "group", _snapshot)
    try:
        assert _poll(first.port, bytes(32)) == (401, b"")
        assert _poll(first.port, key, "invalid") == (401, b"")
        first_response = json.loads(_poll(first.port, key)[1])
        second_response = json.loads(_poll(first.port, key)[1])
        assert first_response["instance_id"] == "group"
        assert first_response["boot_id"] == second_response["boot_id"]
        assert [first_response["sequence"], second_response["sequence"]] == [1, 2]
        assert second_response["status"] == _snapshot()
    finally:
        first.close()
    restarted = NetworkProgressEndpoint(first.port, key.hex(), "group", _snapshot)
    try:
        new_response = json.loads(_poll(restarted.port, key)[1])
        assert new_response["sequence"] == 1
        assert new_response["boot_id"] != first_response["boot_id"]
    finally:
        restarted.close()


def test_network_status_only_exports_predictive_inputs():
    key = bytes(range(32))
    observed = 10**15
    cohort = {
        "cohort_epoch_id": "epoch",
        "cohort_first_queue_enter_ns": observed - 40,
        "cohort_last_queue_enter_ns": observed - 39,
        "cohort_close_ns": observed - 38,
    }
    active = {"start_ns": observed - 3, "forward_id": "f", "request_ids": ["active"]}
    completion = {"kind": "forward_end", "at_ns": observed - 20, "request_ids": ["done"]}
    queue = ["queued-2", "queued-1"]
    full = _snapshot(
        active={**active, **cohort},
        queued_request_ids=queue,
        recent_events=[
            {"kind": "queue_enter", "at_ns": observed - 40},
            {"kind": "cohort_close", "at_ns": observed - 38, "close_ns": observed - 38},
            {"kind": "forward_start", "at_ns": observed - 38, **cohort},
            {**completion, "forward_id": "old-forward", **cohort},
            {"kind": "queue_leave", "at_ns": observed - 20},
            {"kind": "cohort_metadata_rejected", "at_ns": observed - 12},
        ],
    )
    endpoint = NetworkProgressEndpoint(0, key.hex(), "group", lambda: full)
    try:
        assert json.loads(_poll(endpoint.port, key)[1])["status"] == _snapshot(
            active=active, queued_request_ids=queue, recent_events=[completion]
        )
        assert _poll(endpoint.port, key, "elapsed.v2") == (401, b"")
        elapsed = json.loads(_poll(endpoint.port, key, ELAPSED)[1])["status"]
        assert elapsed == {
            "schema": "omni_scheduler_progress.v3",
            "clock_domain": "server_elapsed_ns",
            "active": {"start_age_ns": 3, "forward_id": "f", "request_ids": ["active"]},
            "queued_request_ids": queue,
            "recent_completions": [{"age_ns": 20, "request_ids": ["done"]}],
        }
        assert full["active"]["cohort_close_ns"] == observed - 38
        assert len(full["recent_events"]) == 6
    finally:
        endpoint.close()


@pytest.mark.parametrize("queued", [list(range(1025)), ["x" * (2 * 1024 * 1024)]])
def test_oversized_status_is_dropped_instead_of_looking_idle(queued):
    key = bytes(range(32))
    endpoint = NetworkProgressEndpoint(0, key.hex(), "group", lambda: _snapshot(queued_request_ids=queued))
    try:
        with pytest.raises(http.client.RemoteDisconnected):
            _poll(endpoint.port, key)
    finally:
        endpoint.close()
