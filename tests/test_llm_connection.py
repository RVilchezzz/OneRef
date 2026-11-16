"""Live connection check to the LLM endpoint (HF router by default). Needs network + HF_TOKEN.

    set HF_TOKEN=hf_...            (or ONEREF_LLM_API_KEY)
    python -m pytest tests/test_llm_connection.py -s

Skipped when no key is set, so the GPU-free suite stays offline. Model, URL and key come from
``Settings`` (ONEREF_READER_A_MODEL, ONEREF_LLM_BASE_URL, ...), same init as the pipeline agents.
If the configured model cannot be reached, the same prompt is retried on FALLBACK_MODEL.
"""
import os

import pytest
from dotenv import dotenv_values
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.models import override_allow_model_requests

from inspector.agents.llm import make_role_model
from inspector.config.settings import get_settings

settings = get_settings()

pytestmark = pytest.mark.skipif(settings.llm_api_key in ("", "EMPTY") and settings.llm_provider == "router" and not settings.remote,
                                reason="no HF_TOKEN / ONEREF_LLM_API_KEY set")

IMAGE_URL = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
FALLBACK_MODEL = {**dotenv_values(".env"), **os.environ}.get("ONEREF_FALLBACK_MODEL", "Qwen/Qwen3.8-27B:novita")  # used when the configured model cannot be reached


def ask(prompt):
    """Run ``prompt`` on the configured model, then on FALLBACK_MODEL if that fails."""
    errors = []
    ep = settings.remote.get("reader_a")  # set by ONEREF_USE_GROK / ONEREF_REMOTE
    for name in (settings.reader_a_model, FALLBACK_MODEL):
        # primary: the endpoint's own model (Grok); fallback: same endpoint, FALLBACK_MODEL name
        e = ep if ep and name == settings.reader_a_model else {k: v for k, v in (ep or {}).items() if k != "model"}
        model = make_role_model(settings, name, e)
        name = e.get("model", name)
        agent = Agent(model=model, name="Ping Agent", system_prompt="You are a concise assistant.", retries=3)
        try:
            with override_allow_model_requests(True):  # test_workflow.py disables real requests globally
                res = agent.run_sync(prompt)
            out = res.output
            name = res.all_messages()[-1].model_name or name  # real model (the backup if the primary failed)
        except Exception as e:  # unknown model, provider down, auth...
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        note = f"  (fallback; primary failed: {errors[0]})" if errors else ""
        print(f"\n[{name}] {out}{note}")
        return out
    pytest.fail("no model reachable:\n" + "\n".join(errors))


def test_llm_text_reply():
    out = ask("What is 17 * 23? Answer with the number only, no words.")
    assert "391" in out


def test_llm_image_reply():
    out = ask(["Which famous monument is shown in this image? Answer with its name only, no sentence.", ImageUrl(url=IMAGE_URL)])
    assert "liberty" in out.lower()
