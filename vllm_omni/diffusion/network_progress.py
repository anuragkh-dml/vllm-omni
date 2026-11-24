# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, authenticated scheduler progress on a dedicated loopback port."""

from __future__ import annotations

import hmac
import json
import re
import socket
import socketserver
import threading
import time
import uuid
from collections.abc import Callable

ABSOLUTE = "host_monotonic.v1"
ELAPSED = "elapsed.v3"
MAX_BODY = 2 * 1024 * 1024
MAX_HEADER = 4096
DEADLINE_S = 0.1
NONCE = re.compile(rb"[0-9a-f]{64}")


def _mac(key: bytes, purpose: bytes, nonce: bytes, schema: bytes, body: bytes = b"") -> bytes:
    return (
        hmac.digest(key, b"omni-status.v1:" + purpose + b":" + nonce + b":" + schema + b":" + body, "sha256")
        .hex()
        .encode()
    )


def _predictive_status(status: dict, *, elapsed: bool) -> dict:
    """Project the diagnostic snapshot onto facts consumed by Predictive."""
    observed = status["observed_ns"]

    def age(at_ns: int) -> int:
        duration = observed - at_ns
        if duration < 0:
            raise ValueError("status event after capture")
        return duration

    active = status["active"]
    if active is not None:
        start_ns = active["start_ns"]
        active = {"forward_id": active["forward_id"], "request_ids": active["request_ids"]}
        if elapsed:
            active["start_age_ns"] = age(start_ns)
        else:
            active["start_ns"] = start_ns
    completions = [event for event in status["recent_events"] if event["kind"] == "forward_end"]
    # Allowlist fields so future diagnostic metadata and host-local timestamps
    # cannot leak into network snapshots. Keep scheduler order for prediction.
    if elapsed:
        return {
            "schema": "omni_scheduler_progress.v3",
            "clock_domain": "server_elapsed_ns",
            "active": active,
            "queued_request_ids": status["queued_request_ids"],
            "recent_completions": [
                {"age_ns": age(event["at_ns"]), "request_ids": event["request_ids"]} for event in completions
            ],
        }
    return {
        "schema": "omni_scheduler_progress.v1",
        "clock_domain": "host_monotonic_ns",
        "hostname": status["hostname"],
        "observed_ns": observed,
        "active": active,
        "queued_request_ids": status["queued_request_ids"],
        "recent_events": [
            {"kind": "forward_end", "at_ns": event["at_ns"], "request_ids": event["request_ids"]}
            for event in completions
        ],
    }


class _StatusHandler(socketserver.BaseRequestHandler):
    server: _StatusServer

    def handle(self) -> None:
        self.server.respond(self.request)


class _StatusServer(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 8

    def __init__(self, port: int, key: str, instance: str, snapshot: Callable[[], dict]):
        if not re.fullmatch(r"[0-9a-f]{64}", key) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", instance):
            raise ValueError("status requires a 32-byte lowercase hex key and a bounded instance ID")
        self.key = bytes.fromhex(key)
        self.instance = instance
        self.boot = uuid.uuid4().hex
        self.sequence = 0
        self.snapshot = snapshot
        super().__init__(("127.0.0.1", port), _StatusHandler)

    def _send(self, request: socket.socket, status: bytes, body: bytes, auth: bytes, deadline: float) -> None:
        request.settimeout(max(0.001, deadline - time.monotonic()))
        request.sendall(
            b"HTTP/1.1 "
            + status
            + b"\r\nConnection: close\r\nContent-Type: application/json\r\n"
            + b"Content-Length: "
            + str(len(body)).encode()
            + b"\r\nX-Status-Auth: "
            + auth
            + b"\r\n\r\n"
            + body
        )

    def respond(self, request: socket.socket) -> None:
        deadline = time.monotonic() + DEADLINE_S
        try:
            header = bytearray()
            while b"\r\n\r\n" not in header:
                remaining = deadline - time.monotonic()
                if len(header) == MAX_HEADER or remaining <= 0:
                    return
                request.settimeout(remaining)
                data = request.recv(MAX_HEADER - len(header))
                if not data:
                    return
                header.extend(data)
            head, _, remainder = header.partition(b"\r\n\r\n")
            lines = head.split(b"\r\n")
            if remainder or lines[0] != b"GET /status HTTP/1.1":
                return
            fields = {}
            for line in lines[1:]:
                name, sep, value = line.partition(b":")
                name = bytes(name.lower())
                if not name or not sep or name in fields:
                    return
                fields[name] = bytes(value.strip())
            if fields.get(b"content-length", b"0") != b"0" or b"transfer-encoding" in fields:
                return
            nonce = fields.get(b"x-status-nonce", b"")
            schema = fields.get(b"x-status-schema", b"")
            auth = fields.get(b"x-status-auth", b"")
            if (
                not NONCE.fullmatch(nonce)
                or schema not in (ABSOLUTE.encode(), ELAPSED.encode())
                or not hmac.compare_digest(auth, _mac(self.key, b"request", nonce, schema))
            ):
                self._send(request, b"401 Unauthorized", b"", b"", deadline)
                return
            self.sequence += 1
            snapshot = self.snapshot()
            # An oversized state is unavailable. Dropping unknown queued or
            # active work would publish a misleading capacity observation.
            if (
                len(snapshot.get("queued_request_ids", ())) > 1024
                or len((snapshot["active"] or {}).get("request_ids", ())) > 1024
                or len(snapshot.get("recent_events", ())) > 256
            ):
                return
            snapshot = _predictive_status(snapshot, elapsed=schema == ELAPSED.encode())
            envelope = {
                "instance_id": self.instance,
                "boot_id": self.boot,
                "sequence": self.sequence,
                "status": snapshot,
            }
            body = bytearray()
            for part in json.JSONEncoder(separators=(",", ":"), allow_nan=False).iterencode(envelope):
                if len(part) > MAX_BODY - len(body):
                    return
                body.extend(part.encode())
                if len(body) > MAX_BODY:
                    return
            if time.monotonic() >= deadline:
                return
            body = bytes(body)
            self._send(request, b"200 OK", body, _mac(self.key, b"response", nonce, schema, body), deadline)
        except (OSError, KeyError, TypeError, ValueError):
            # An interrupted poll cannot block the scheduler or expose partial status.
            return


class NetworkProgressEndpoint:
    """The scheduler only provides a synchronized snapshot. One worker owns I/O."""

    def __init__(self, port: int, key: str, instance: str, snapshot: Callable[[], dict]):
        self._server = _StatusServer(port, key, instance, snapshot)
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.05,), daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
