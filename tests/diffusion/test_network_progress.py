# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import http.client
import json
import secrets

from vllm_omni.diffusion.network_progress import ABSOLUTE, NetworkProgressEndpoint, _mac


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
