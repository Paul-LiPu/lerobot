import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from typing import Any


class ChromeTraceRecorder:
    def __init__(self, output_path: str | Path | None, *, process_name: str):
        self.output_path = Path(output_path) if output_path is not None else None
        self.process_name = process_name
        self._pid = os.getpid()
        self._start_ns = perf_counter_ns()
        self._events: list[dict[str, Any]] = []
        self._lock = Lock()
        self._closed = False
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._record_metadata("process_name", {"name": process_name}, tid=0)

    @property
    def enabled(self) -> bool:
        return self.output_path is not None

    def _ts_us(self, ns: int) -> float:
        return (ns - self._start_ns) / 1_000

    def _append_event(self, event: dict[str, Any]) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            self._events.append(event)

    def _record_metadata(self, name: str, args: dict[str, Any], *, tid: int) -> None:
        self._append_event(
            {
                "name": name,
                "ph": "M",
                "pid": self._pid,
                "tid": tid,
                "ts": 0,
                "args": args,
            }
        )

    def set_thread_name(self, name: str, *, tid: int | None = None) -> None:
        if not self.enabled:
            return
        self._record_metadata("thread_name", {"name": name}, tid=get_ident() if tid is None else tid)

    @contextmanager
    def span(self, name: str, *, args: dict[str, Any] | None = None, category: str = "robot"):
        if not self.enabled or self._closed:
            yield
            return

        start_ns = perf_counter_ns()
        tid = get_ident()
        try:
            yield
        finally:
            end_ns = perf_counter_ns()
            self._append_event(
                {
                    "name": name,
                    "cat": category,
                    "ph": "X",
                    "pid": self._pid,
                    "tid": tid,
                    "ts": self._ts_us(start_ns),
                    "dur": (end_ns - start_ns) / 1_000,
                    "args": args or {},
                }
            )

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            payload = {"traceEvents": self._events, "displayTimeUnit": "ms"}
            self.output_path.write_text(json.dumps(payload), encoding="utf-8")
            self._closed = True
