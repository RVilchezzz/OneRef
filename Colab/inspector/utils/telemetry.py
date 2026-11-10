"""Structured telemetry: JSONL per run (always on) + optional Logfire/OpenTelemetry.

Each ``span`` writes one JSON line with: run, image_id, agent, prompt(s), raw response(s),
latency (ms), GPU memory (MB), tokens in/out and any extra fields the agent adds
(e.g. the final verdict).
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _gpu_mem_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:  # torch missing or CUDA broken: telemetry must never crash a run
        pass
    return None


class Telemetry:
    def __init__(self, jsonl_path: str | Path, run: str = "run", use_logfire: bool = False):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run
        self._lock = threading.Lock()
        self._logfire: Any = None
        if use_logfire:
            try:  # optional dependency; no account needed unless this flag is set
                import logfire

                logfire.configure(send_to_logfire="if-token-present")
                logfire.instrument_pydantic_ai()  # traces every agent run, as in PasteurAIze
                self._logfire = logfire
            except Exception as e:  # noqa: BLE001
                self.write({"event": "logfire_unavailable", "error": repr(e)})

    def write(self, record: dict[str, Any]) -> None:
        record = {"run": self.run, "ts": time.time(), **record}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def span(self, agent: str, image_id: str) -> Iterator[dict[str, Any]]:
        """Yield a dict the agent fills (``prompt``, ``raw``, ``tokens_in``, ``tokens_out``...).

        The record is written even when the agent raises (with ``error``), then re-raised.
        """
        rec: dict[str, Any] = {"agent": agent, "image_id": image_id}
        cm = self._logfire.span("{agent} {image_id}", agent=agent, image_id=image_id) if self._logfire else None
        if cm:
            cm.__enter__()
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as e:
            rec["error"] = repr(e)
            raise
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            rec["gpu_mem_mb"] = _gpu_mem_mb()
            self.write(rec)
            if cm:
                cm.__exit__(None, None, None)
