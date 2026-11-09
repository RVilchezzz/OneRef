"""Global configuration (Pydantic Settings).

Every value can be overridden by an env var prefixed ``ONEREF_`` (e.g.
``ONEREF_DEVICE=cpu``) or by a ``.env`` file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]  # OneRef/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONEREF_", env_file=".env", extra="ignore")

    # ---- paths -------------------------------------------------------------
    mmad_root: Path = ROOT / "data" / "MMAD"          # images (gitignored)
    mmad_json: Path = ROOT / "data" / "MMAD" / "mmad.json"
    splits_dir: Path = ROOT / "data" / "splits"
    outputs_dir: Path = ROOT / "outputs"              # per-agent cache + JSONL (gitignored)

    # ---- hardware / pass budget -------------------------------------------
    device: str = "cuda"                              # falls back to cpu if CUDA is unavailable
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_in_4bit: bool = False                        # True for 6 GB GPU / T4
    max_models_on_gpu: int = 1                        # pass-based execution: one model at a time

    # ---- models ------------------------------------------------------------
    expert_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    expert_fallback_model: str = "facebook/dinov2-base"
    reader_a_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    reader_b_model: str = "OpenGVLab/InternVL3-8B-hf"
    reader_backend: Literal["hf", "vllm"] = "hf"
    max_new_tokens: int = 256
    json_retries: int = 1                             # one retry on malformed JSON

    # ---- expert calibration (tune on dev only) ----------------------------
    anomaly_norm_low: float = 0.2                     # cosine distance mapped to score 0
    anomaly_norm_high: float = 0.8                    # cosine distance mapped to score 1
    expert_flag_threshold: float = 0.5                # score >= threshold => "defect" vote

    # ---- protocol ----------------------------------------------------------
    dev_fraction: float = Field(0.10, gt=0, lt=1)
    seed: int = 0
    max_reopen: int = 1                               # MAST: no unbounded loops

    # ---- aggregation weights: {agent: {subtask: weight}}, default 1.0 -----
    default_weight: float = 1.0
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)  # fixed on dev, e.g. {"reader_a": {"defect_localization": 1.3}}
    expert_weight_discrimination: float = 1.0

    # ---- critic ------------------------------------------------------------
    # Rule "discrimination=No but the image has defect-localization questions" depends on the
    # *presence* of questions, which reveals the label in MMAD. Off by default (strict protocol).
    critic_use_question_structure: bool = False

    # ---- telemetry ---------------------------------------------------------
    use_logfire: bool = False                         # optional; JSONL is always on


def get_settings() -> Settings:
    return Settings()
