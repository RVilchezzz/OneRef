"""Global configuration (Pydantic Settings).

Every value can be overridden by an env var prefixed ``ONEREF_`` (e.g.
``ONEREF_DEVICE=cpu``) or by a ``.env`` file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]  # OneRef/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONEREF_", env_file=".env", extra="ignore", populate_by_name=True)

    # ---- paths -------------------------------------------------------------
    mmad_root: Path = ROOT / "data" / "MMAD"          # images (gitignored)
    mmad_json: Path = ROOT / "data" / "MMAD" / "mmad.json"
    splits_dir: Path = ROOT / "data" / "splits"
    outputs_dir: Path = ROOT / "outputs"              # per-agent cache + JSONL (gitignored)

    # ---- hardware / pass budget -------------------------------------------
    device: str = "cuda"                              # falls back to cpu if CUDA is unavailable
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"   # visual expert only
    max_models_on_gpu: int = 1                        # pass-based execution: one model at a time

    # ---- models ------------------------------------------------------------
    expert_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    expert_fallback_model: str = "facebook/dinov2-base"
    reader_a_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    reader_b_model: str = "OpenGVLab/InternVL3-8B"
    critic_model: str = "Qwen/Qwen3-VL-8B-Instruct"  # same weights as reader A, critic prompt
    max_new_tokens: int = 256
    json_retries: int = 1                             # pydantic-ai output retries on bad JSON / option

    # ---- LLM endpoint (OpenAI-compatible) ----------------------------------
    # "router": Hugging Face router (https://router.huggingface.co/v1), key = HF_TOKEN, no local server.
    #           Model ids are HF ids, optionally with a provider suffix ("Qwen/Qwen3-VL-8B-Instruct:novita").
    # "vllm":   a local/Colab vLLM server (see vllm_* below); set ONEREF_LLM_BASE_URL=http://localhost:8000/v1
    #           and ONEREF_LLM_API_KEY=EMPTY.
    llm_provider: Literal["router", "vllm"] = "router"
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_api_key: str = Field("EMPTY", validation_alias=AliasChoices("ONEREF_LLM_API_KEY", "HF_TOKEN"))
    # True: each GPU pass starts `vllm serve <model>` and stops it at the end of the pass.
    # False: a server is already running; the pass only checks it serves the right model.
    vllm_autostart: bool = False
    vllm_args: list[str] = [
        "--max-model-len", "8192", "--gpu-memory-utilization", "0.9", "--trust-remote-code",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",  # needed by the critic's tool
    ]
    vllm_startup_timeout_s: int = 900
    # Hosted API per agent, e.g. ONEREF_REMOTE='{"reader_a":{"base_url":"https://api.x.ai/v1","api_key":"...","model":"grok-..."}}'
    # Roles: reader_a, reader_b, critic. A listed role skips vLLM entirely (nothing on the GPU).
    remote: dict[str, dict[str, str]] = {}

    # A role with a `remote` endpoint falls back to the default llm_* endpoint on API errors (quota...).
    backup_enabled: bool = True
    llm_api_keys: str = ""         # extra keys (commas/spaces) for the default endpoint (e.g. Gemini): tried after llm_api_key

    # ---- Grok (xAI, OpenAI-compatible) -------------------------------------
    # ONEREF_USE_GROK=true fills `remote` for the three roles (explicit ONEREF_REMOTE entries win).
    # Hosted: leaves the strict local <=8B protocol, report as upper bound. Model ids are placeholders.
    use_grok: bool = False
    grok_base_url: str = "https://api.x.ai/v1"
    grok_api_key: str = Field("", validation_alias=AliasChoices("ONEREF_GROK_API_KEY", "XAI_API_KEY"))
    grok_api_keys: str = ""        # extra Groq keys, separated by commas/spaces: tried in order when a key hits its quota
    grok_reader_a_model: str = "grok-4"
    grok_reader_b_model: str = "grok-4"
    grok_critic_model: str = "grok-4"

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
    use_llm_critic: bool = True                       # False = rules-only critic (ablation, no GPU pass)

    # ---- telemetry ---------------------------------------------------------
    use_logfire: bool = False                         # optional; JSONL is always on

    @staticmethod
    def split_keys(raw: str) -> list[str]:
        return [k for k in re.split(r"[,;\s]+", raw) if k]

    @model_validator(mode="after")
    def _grok_remote(self) -> "Settings":
        if not self.use_grok:
            return self
        if not self.grok_api_key:
            raise ValueError("ONEREF_USE_GROK=true needs ONEREF_GROK_API_KEY (or XAI_API_KEY)")
        for role, model in (("reader_a", self.grok_reader_a_model), ("reader_b", self.grok_reader_b_model), ("critic", self.grok_critic_model)):
            self.remote.setdefault(role, {"base_url": self.grok_base_url, "api_key": self.grok_api_key, "model": model})
        return self


def get_settings() -> Settings:
    return Settings()
