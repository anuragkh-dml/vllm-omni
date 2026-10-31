"""Optional host-monotonic scheduler status for benchmark admission.

Enabled only when VLLM_OMNI_BENCH_STATUS_PATH names a writable local file.
The scheduler only replaces an in-memory state; file I/O runs in a daemon.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
import socket
import threading
import time


class ProgressFeed:
    def __init__(self, path: str):
        self.path = Path(path)
        self.hostname = socket.gethostname()
        self._lock = threading.Lock()
        self._active: dict | None = None
        self._queued: tuple[str, ...] = ()
        self._events: deque[dict] = deque(maxlen=256)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="omni-progress-feed", daemon=True)
        self._thread.start()

    def publish(self, active: dict | None, queued: tuple[str, ...]) -> None:
        with self._lock:
            at_ns = time.monotonic_ns()
            previous_active = set(self._active["request_ids"]) if self._active else set()
            current_active = set(active["request_ids"]) if active else set()
            previous_queued = set(self._queued)
            current_queued = set(queued)
            for request_id in sorted(current_queued - previous_queued - previous_active):
                self._events.append({"at_ns": at_ns, "kind": "queue_enter", "request_id": request_id})
            for request_id in sorted(previous_queued - current_queued):
                self._events.append({"at_ns": at_ns, "kind": "queue_leave", "request_id": request_id})
            if self._active is not None and self._active != active:
                self._events.append({"at_ns": at_ns, "kind": "forward_end",
                                     "forward_id": self._active["forward_id"],
                                     "request_ids": self._active["request_ids"]})
            if active is not None and self._active != active:
                self._events.append({"at_ns": at_ns, "kind": "forward_start",
                                     "forward_id": active["forward_id"],
                                     "request_ids": active["request_ids"]})
            self._active = active
            self._queued = queued
        self._wake.set()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            with self._lock:
                active = self._active
                queued = self._queued
                events = list(self._events)
            observed_ns = time.monotonic_ns()
            payload = {"schema": "omni_scheduler_progress.v1",
                       "clock_domain": "host_monotonic_ns",
                       "hostname": self.hostname,
                       "observed_ns": observed_ns,
                       "active": active,
                       "queued_request_ids": queued,
                       "recent_events": events}
            temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                os.replace(temporary, self.path)
            except OSError:
                # The reader will treat a missing or stale feed as a fallback.
                pass
            self._wake.wait(0.01)
            self._wake.clear()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1)
