"""Deterministic workflow definition: an explicit, ordered list of typed steps (no free-form loops).

    expert -> reader_a -> reader_b -> critic -> [reader_a_reopen -> reader_b_reopen] -> aggregator

The only "loop" is the reopen pair, which is a straight-line pass that runs only for images the
critic flagged, and the critic grants it at most ``max_reopen`` (=1) times per image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inspector.agents.aggregator import Aggregator
from inspector.agents.base import BaseAgent
from inspector.agents.critic_verifier import CriticVerifier
from inspector.config.settings import Settings


@dataclass(frozen=True)
class Step:
    name: str                       # also the cache sub-folder name
    make_agent: Callable[[], BaseAgent]  # built lazily: nothing heavy is created until the pass starts


def build_steps(settings: Settings, run_dir: Path, mock: bool = False) -> list[Step]:
    if mock:
        from inspector.agents.mock import MockExpert, MockReader

        expert: Callable[[], BaseAgent] = MockExpert
        reader = lambda name, reopen: (lambda: MockReader(settings, name, reopen))  # noqa: E731
    else:
        from inspector.agents.reader_mllm import ReaderMLLM
        from inspector.agents.visual_expert import VisualExpert

        expert = lambda: VisualExpert(settings, run_dir)  # noqa: E731
        models = {"reader_a": settings.reader_a_model, "reader_b": settings.reader_b_model}
        reader = lambda name, reopen: (lambda: ReaderMLLM(settings, models[name.removesuffix("_reopen")], name, reopen))  # noqa: E731

    return [
        Step("1_expert", expert),
        Step("2_reader_a", reader("reader_a", False)),
        Step("3_reader_b", reader("reader_b", False)),
        Step("4_critic", lambda: CriticVerifier(settings)),
        Step("5_reader_a_reopen", reader("reader_a_reopen", True)),
        Step("6_reader_b_reopen", reader("reader_b_reopen", True)),
        Step("7_aggregator", lambda: Aggregator(settings)),
    ]
