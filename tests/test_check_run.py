"""Gate between the 50-image probe and the full dev run, and the 8B Colab config."""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_run import report  # noqa: E402
from inspector.config.settings import Settings  # noqa: E402

STEPS = ["1_expert", "2_reader_a", "3_reader_b", "4_critic", "6b_judge", "7_aggregator"]
AGENT = {"1_expert": "visual_expert", "2_reader_a": "reader_a", "3_reader_b": "reader_b",
         "4_critic": "critic", "6b_judge": "judge", "7_aggregator": "aggregator"}


def telemetry(bad_reader_calls=0, expert="facebook/dinov3-vitb16", llm=True, final=True):
    """One image, 100 questions, every pass 10 s."""
    ev, t = [], 0.0
    for step in STEPS:
        ev.append({"event": "pass_start", "ts": t, "step": step, "cached": 0, "to_run": 1})
        t += 10
        rec = {"agent": AGENT[step], "image_id": "i", "ts": t, "latency_ms": 8000}
        if step == "1_expert":
            rec["model"] = expert
        if step.endswith(("reader_a", "reader_b")):
            rec["calls"] = [{"question_id": str(i), **({"error": "x"} if i < bad_reader_calls else {})} for i in range(100)]
        if step == "4_critic":
            rec["llm" if llm else "llm_error"] = "x"
        if step == "6b_judge":
            rec["answers"] = {"0": "A"}
        if step == "7_aggregator":
            rec["final"] = {str(i): "A" for i in range(100)}
        ev += [rec, {"event": "pass_end", "ts": t, "step": step}]
    if final:
        ev.append({"event": "final_metrics", "ts": t, "mean": 0.6})
    return ev


def test_clean_run_passes_and_projects_duration():
    r = report(telemetry(), project=835)
    assert r["failures"] == []
    assert abs(r["hours"] - (6 * 2 + 835 * 6 * 8) / 3600) < 1e-6  # 2 s start-up + 8 s per image, per pass


def test_each_threshold_fails():
    assert any("reader_a" in f for f in report(telemetry(bad_reader_calls=5))["failures"])
    assert not report(telemetry(bad_reader_calls=4))["failures"]
    assert any("critic" in f for f in report(telemetry(llm=False))["failures"])
    assert any("expert" in f for f in report(telemetry(expert="facebook/dinov2-base"))["failures"])
    assert any("final" in f for f in report(telemetry(final=False))["failures"])
    unfinished = [e for e in telemetry() if not (e.get("event") == "pass_end" and e["step"] == "6b_judge")]
    assert any("6b_judge" in f for f in report(unfinished)["failures"])


def test_8b_template_loads_in_settings(monkeypatch):
    text = (ROOT / "configs" / "2026-11-19_8B_colab_template.env").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):  # same parsing as notebook cell 5
            k, v = line.split("=", 1)
            monkeypatch.setenv(k, shlex.split(v)[0] if v else "")
    s = Settings(_env_file=None)
    assert (s.reader_a_model, s.reader_b_model) == ("Qwen/Qwen3-VL-8B-Instruct", "OpenGVLab/InternVL3_5-8B")
    assert s.critic_model == s.judge_model == "google/gemma-4-E4B-it"
    assert s.critic_use_tool is False
    assert s.critic_extra_body == s.judge_extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert json.dumps(s.critic_extra_body)
