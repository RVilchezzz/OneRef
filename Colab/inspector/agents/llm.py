"""Shared LLM plumbing for the pydantic-ai agents (same pattern as PasteurAIze).

* ``make_model``: an ``OpenAIChatModel`` pointed at an OpenAI-compatible endpoint (vLLM, Ollama...).
* ``image_part``: an image file as ``BinaryContent`` for a multimodal ``agent.run([...])``.
* ``VLLMServer``: one ``vllm serve <model>`` per GPU pass, so only one model is ever on the GPU.
* ``LLMAgent``: base for the reader and critic agents: builds the pydantic-ai ``Agent`` on
  ``load()``, owns one event loop for the whole pass, and stops the server on ``unload()``.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import subprocess
import time
import urllib.request
from abc import abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from inspector.agents.base import BaseAgent
from inspector.config.settings import Settings
from inspector.schemas.state import InspectionState


def make_model(s: Settings, model_name: str, ep: dict[str, str] | None = None) -> OpenAIChatModel:
    ep = ep or {}
    return OpenAIChatModel(
        ep.get("model", model_name),
        provider=OpenAIProvider(base_url=ep.get("base_url", s.llm_base_url), api_key=ep.get("api_key", s.llm_api_key)),
    )


def make_role_model(s: Settings, model_name: str, ep: dict[str, str] | None = None) -> Model:
    """``make_model`` plus failover (pydantic-ai ``FallbackModel``, any API error such as 429/402 quota, 5xx):
    1. the role's hosted endpoint (e.g. Groq) with its extra keys ``grok_api_keys``;
    2. the default ``llm_*`` endpoint (e.g. Gemini) with ``llm_api_key`` then ``llm_api_keys``.
    Without a hosted endpoint, only the extra ``llm_api_keys`` of the default endpoint are added."""
    chain = [make_model(s, model_name, ep)]
    if ep and s.use_grok and ep.get("base_url") == s.grok_base_url:
        chain += [make_model(s, model_name, {**ep, "api_key": k}) for k in s.split_keys(s.grok_api_keys)]
    if s.backup_enabled and s.llm_provider == "router":
        extra = s.split_keys(s.llm_api_keys)
        keys = ([s.llm_api_key] if ep and s.llm_api_key not in ("", "EMPTY") else []) + extra
        chain += [make_model(s, model_name, {"api_key": k}) for k in keys if k != s.llm_api_key or ep]
    return chain[0] if len(chain) == 1 else FallbackModel(*chain)


def image_part(path: str) -> BinaryContent:
    media = mimetypes.guess_type(path)[0] or "image/png"
    return BinaryContent(data=Path(path).read_bytes(), media_type=media)


def response_texts(messages: list[ModelMessage]) -> list[str]:
    """Raw text of every model response (for the JSONL log; images are never logged)."""
    return [p.content for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, TextPart)]


def served_models(s: Settings) -> list[str] | None:
    """Model ids served at ``llm_base_url``, or None if nothing answers."""
    req = urllib.request.Request(s.llm_base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {s.llm_api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except OSError:
        return None


class VLLMServer:
    """Starts (if ``vllm_autostart``) or checks the endpoint for one model; ``stop()`` frees the GPU."""

    def __init__(self, s: Settings, model_name: str):
        self.s, self.model_name = s, model_name
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not self.s.vllm_autostart:
            models = served_models(self.s)
            if models is None:
                raise RuntimeError(f"No LLM server at {self.s.llm_base_url}. Start `vllm serve {self.model_name}` or set ONEREF_VLLM_AUTOSTART=true.")
            if self.model_name not in models:
                raise RuntimeError(f"Server at {self.s.llm_base_url} serves {models}, not {self.model_name!r}.")
            return
        port = str(urlparse(self.s.llm_base_url).port or 8000)
        self.proc = subprocess.Popen(["vllm", "serve", self.model_name, "--port", port, *self.s.vllm_args])
        deadline = time.monotonic() + self.s.vllm_startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve {self.model_name} exited with code {self.proc.returncode}")
            if self.model_name in (served_models(self.s) or []):
                return
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"vllm serve {self.model_name} not ready after {self.s.vllm_startup_timeout_s}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class LLMAgent(BaseAgent):
    """An agent backed by a pydantic-ai ``Agent``. ``model`` injection = no server (mock / tests)."""

    def __init__(self, settings: Settings, model_name: str, model: Model | None = None):
        self.s = settings
        self.model_name = model_name
        self._model = model
        self.agent: Agent | None = None
        self.server: VLLMServer | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    @abstractmethod
    def build(self, model: Model) -> Agent:
        """Create the pydantic-ai Agent (prompt, output_type, tools, validators)."""

    def enabled(self) -> bool:
        return True

    def load(self) -> None:
        if not self.enabled():
            return
        model = self._model
        if model is None:
            ep = self.s.remote.get(self.name.removesuffix("_reopen"))  # hosted API: no local server
            if ep is None and self.s.llm_provider == "vllm":
                self.server = VLLMServer(self.s, self.model_name)
                self.server.start()
            model = make_role_model(self.s, self.model_name, ep)
        self.agent = self.build(model)
        self.loop = asyncio.new_event_loop()  # one loop per pass: the HTTP client stays bound to it

    def unload(self) -> None:
        if self.server:
            self.server.stop()
        if self.loop:
            self.loop.close()
        self.agent = self.server = self.loop = None
        super().unload()

    def image_parts(self, state: InspectionState) -> list[BinaryContent]:
        """Query, normal reference (random_templates[0]) and the expert's zoomed crop."""
        ev = state.visual_evidence
        paths = [state.query_image_path, state.reference_image_path] + ([ev.crop_path] if ev else [])
        return [image_part(p) for p in paths]

    def run_async(self, coro: Any) -> Any:
        assert self.loop is not None, "load() must be called before run()"
        return self.loop.run_until_complete(coro)
