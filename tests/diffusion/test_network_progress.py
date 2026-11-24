# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import http.client
import json
import secrets

import pytest

from vllm_omni.diffusion.network_progress import ABSOLUTE, ELAPSED, NetworkProgressEndpoint, _mac


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


def test_status_authentication_order_and_server_restart():
    key = bytes(range(32))
    first = NetworkProgressEndpoint(0, key.hex(), "group", lambda: {"active": None})
    try:
        assert _poll(first.port, bytes(32)) == (401, b"")
        assert _poll(first.port, key, "invalid") == (401, b"")
        first_response = json.loads(_poll(first.port, key)[1])
        second_response = json.loads(_poll(first.port, key)[1])
        assert first_response["instance_id"] == "group"
        assert first_response["boot_id"] == second_response["boot_id"]
        assert [first_response["sequence"], second_response["sequence"]] == [1, 2]
        assert second_response["status"] == {"active": None}
    finally:
        first.close()
    restarted = NetworkProgressEndpoint(first.port, key.hex(), "group", lambda: {"active": None})
    try:
        new_response = json.loads(_poll(restarted.port, key)[1])
        assert new_response["sequence"] == 1
        assert new_response["boot_id"] != first_response["boot_id"]
    finally:
        restarted.close()


def test_elapsed_status_preserves_fields_without_exporting_absolute_times():
    key = bytes(range(32))
    full = {
        "schema": "omni_scheduler_progress.v1",
        "clock_domain": "host_monotonic_ns",
        "hostname": "a-different-host",
        "observed_ns": 10**15,
        "active": {"start_ns": 10**15 - 3, "forward_id": "f", "request_ids": ["active"]},
        "queued_request_ids": ["queued"],
        "recent_events": [
            {"kind": "cohort_close", "at_ns": 10**15 - 20, "close_ns": 10**15 - 20, "epoch_id": "test"},
        ],
    }
    endpoint = NetworkProgressEndpoint(0, key.hex(), "group", lambda: full)
    try:
        assert json.loads(_poll(endpoint.port, key)[1])["status"] == full
        elapsed = json.loads(_poll(endpoint.port, key, ELAPSED)[1])["status"]
        assert elapsed == {
            "schema": "omni_scheduler_progress.v2",
            "clock_domain": "server_elapsed_ns",
            "hostname": "a-different-host",
            "active": {"start_age_ns": 3, "forward_id": "f", "request_ids": ["active"]},
            "queued_request_ids": ["queued"],
            "recent_events": [{"kind": "cohort_close", "age_ns": 20, "close_age_ns": 20, "epoch_id": "test"}],
        }
    finally:
        endpoint.close()


@pytest.mark.parametrize("queued", [list(range(1025)), ["x" * (2 * 1024 * 1024)]])
def test_oversized_status_is_dropped_instead_of_looking_idle(queued):
    key = bytes(range(32))
    endpoint = NetworkProgressEndpoint(0, key.hex(), "group", lambda: {"active": None, "queued_request_ids": queued})
    try:
        with pytest.raises(http.client.RemoteDisconnected):
            _poll(endpoint.port, key)
    finally:
        endpoint.close()
