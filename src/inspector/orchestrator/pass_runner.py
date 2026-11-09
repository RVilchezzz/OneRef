"""Pass-based runner: each step runs over the WHOLE split, then frees the GPU.

Peak VRAM = one model (an 8B MLLM, or DINOv3), never two. After every pass the full
``InspectionState`` of each image is cached to ``<run_dir>/<step>/<image_id>.json``, so a crashed or
disconnected Colab session resumes without recomputing finished images, and baselines can reuse
the readers' cached answers.
"""
from __future__ import annotations

from pathlib import Path

from inspector.agents.base import free_gpu
from inspector.orchestrator.graph import Step
from inspector.schemas.state import InspectionState
from inspector.utils.telemetry import Telemetry


class PassRunner:
    def __init__(self, run_dir: Path, tel: Telemetry):
        self.run_dir = Path(run_dir)
        self.tel = tel

    def _cache(self, step: Step, image_id: str) -> Path:
        return self.run_dir / step.name / (image_id.replace("/", "__") + ".json")

    def run(self, steps: list[Step], states: list[InspectionState]) -> list[InspectionState]:
        for step in steps:
            states = self._run_step(step, states)
        return states

    def _run_step(self, step: Step, states: list[InspectionState]) -> list[InspectionState]:
        done: dict[str, InspectionState] = {}
        pending: list[InspectionState] = []
        for s in states:
            p = self._cache(step, s.image_id)
            if p.exists():
                done[s.image_id] = InspectionState.model_validate_json(p.read_text(encoding="utf-8"))
            else:
                pending.append(s)

        agent = step.make_agent()
        todo = [s for s in pending if agent.applies(s)]
        self.tel.write({"event": "pass_start", "step": step.name, "cached": len(done), "to_run": len(todo)})
        if todo:  # skip loading weights entirely when nothing needs this model
            agent.load()
            try:
                for s in todo:
                    done[s.image_id] = agent.run(s, self.tel)
            finally:  # always release the GPU, even on error, before the next pass
                agent.unload()
        del agent
        free_gpu()

        for s in pending:
            res = done.setdefault(s.image_id, s)  # images the agent skipped pass through unchanged
            p = self._cache(step, s.image_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(res.model_dump_json(), encoding="utf-8")
        self.tel.write({"event": "pass_end", "step": step.name})
        return [done[s.image_id] for s in states]
