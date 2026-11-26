"""GPU-free tests: schemas, strict 1-shot, critic/reopen guard, aggregator, metrics, pass runner."""
import json

import pytest
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from inspector.agents.aggregator import Aggregator
from inspector.agents.conflict_judge import ConflictJudge, conflicts
from inspector.agents.critic_verifier import CriticVerifier, cell_in_text
from inspector.agents.llm import make_model, make_role_model
from inspector.agents.llm_aggregator import LLMAggregator
from inspector.agents.mock import MockExpert
from inspector.agents.reader_mllm import ReaderMLLM
from inspector.config.settings import Settings
from inspector.orchestrator.graph import Step, build_steps
from inspector.orchestrator.pass_runner import PassRunner
from inspector.schemas.agent_outputs import ReaderOutput, VisualExpertOutput
from inspector.schemas.mmad import Question, Sample, SubTask, load_mmad, parse_subtask
from inspector.schemas.state import InspectionState
from inspector.utils import metrics
from inspector.utils.splits import make_split, sample_per_source, stratified_subsample
from inspector.utils.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Settings reads os.environ: ignore ONEREF_* / keys set by a notebook profile or the shell."""
    import os
    for k in [k for k in os.environ if k.startswith("ONEREF_")] + ["HF_TOKEN", "XAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)

models.ALLOW_MODEL_REQUESTS = False  # tests never call a real LLM endpoint

DISC = {"A": "Yes", "B": "No"}
LOC = {"A": "top left", "B": "top right", "C": "center", "D": "bottom right"}


def mk_questions(img="MVTec/bottle/x"):
    return [
        Question(id=f"{img}#0", text="Is there any defect?", options=DISC, answer="A", subtask=SubTask.ANOMALY_DISCRIMINATION),
        Question(id=f"{img}#1", text="Where is it?", options=LOC, answer="B", subtask=SubTask.DEFECT_LOCALIZATION),
    ]


def mk_sample(i=0, defective=True):
    img = f"MVTec/bottle/{i}"
    return Sample(image_id=img, image_path=img + ".png", is_defective=defective,
                  templates=["ref0.png", "ref1.png"], questions=mk_questions(img))


def mk_state(i=0):
    return InspectionState.from_sample(mk_sample(i))


def ans(qid, letter, conf=0.9):
    return ReaderOutput(question_id=qid, selected_option=letter, confidence=conf)


def expert(score=0.9, region="top right"):
    return VisualExpertOutput(heatmap_path="h", candidate_region=region, confidence_score=score, crop_path="c")


# ---- schemas -----------------------------------------------------------------
def test_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        ReaderOutput(question_id="q", selected_option="E", confidence=0.5)
    with pytest.raises(ValidationError):
        ReaderOutput(question_id="q", selected_option="A", confidence=1.5)
    with pytest.raises(ValidationError):
        VisualExpertOutput(heatmap_path="h", candidate_region="nowhere", confidence_score=0.5, crop_path="c")
    with pytest.raises(ValidationError):
        Question(id="q", text="t", options=DISC, answer="C", subtask=SubTask.ANOMALY_DISCRIMINATION)


def test_strict_one_shot_reference():
    assert mk_sample().reference == "ref0.png"
    assert mk_state().reference_image_path == "ref0.png"


def test_load_mmad_ignores_similar_templates(tmp_path):
    entry = {"conversation": [{"Question": "Is there any defect?", "Options": DISC, "Answer": "B", "type": "Anomaly Discrimination"}],
             "random_templates": ["r0.png", "r1.png"], "similar_templates": ["s0.png"], "mask_path": None}
    p = tmp_path / "mmad.json"
    p.write_text(json.dumps({"MVTec/bottle/test/good/000.png": entry}))
    (s,) = load_mmad(p)
    assert s.reference == "r0.png" and not hasattr(s, "similar_templates") and not s.is_defective


def test_parse_subtask_accepts_every_type_in_mmad_json():
    # the 9 "type" strings found in mmad.json (corrected revision, Oct 22 2026)
    raw = {"Anomaly Detection": SubTask.ANOMALY_DISCRIMINATION, "Defect Classification": SubTask.DEFECT_CLASSIFICATION,
           "Defect Localization": SubTask.DEFECT_LOCALIZATION, "Defect Description": SubTask.DEFECT_DESCRIPTION,
           "Defect Analysis": SubTask.DEFECT_ANALYSIS, "Object Classification": SubTask.OBJECT_CLASSIFICATION,
           "Object Analysis": SubTask.OBJECT_ANALYSIS, "Object Structure": SubTask.OBJECT_ANALYSIS,
           "Object Details": SubTask.OBJECT_ANALYSIS}
    assert {k: parse_subtask(k) for k in raw} == raw


def scripted(*replies):
    """FunctionModel replying with the given texts / tool calls in order; records the calls."""
    calls = []

    def fn(messages, info):
        calls.append(messages)
        r = replies[min(len(calls), len(replies)) - 1]
        part = ToolCallPart(tool_name=r[1], args={}) if isinstance(r, tuple) else TextPart(r)
        return ModelResponse(parts=[part])

    return FunctionModel(fn), calls


def run_reader(model, state):
    r = ReaderMLLM(Settings(json_retries=1), "m", "reader_a", model=model)
    r.image_parts = lambda st: []
    r.load()
    try:
        return r._run(state, rec := {}), rec
    finally:
        r.unload()


def test_reader_agent_retries_on_option_not_offered():
    st = mk_state()
    st.questions = st.questions[:1]
    # "C" is not offered on a yes/no question -> output_validator raises ModelRetry -> second answer kept
    model, calls = scripted('{"selected_option": "C", "confidence": 0.7}', '{"selected_option": "B", "confidence": 0.7}')
    st, rec = run_reader(model, st)
    a = st.reader_a_responses[st.questions[0].id]
    assert a.selected_option == "B" and a.question_id == st.questions[0].id
    assert len(calls) == 2 and rec["calls"][0]["raw"] and "tokens_in" in rec


@pytest.mark.parametrize("hint", ["full", "none"])
def test_reader_hint_none_hides_expert_text_and_crop(monkeypatch, hint):
    import inspector.agents.llm as llm
    monkeypatch.setattr(llm, "image_part", lambda path, max_side=None: path)  # no image files needed
    st = mk_state()
    st.visual_evidence = expert()
    r = ReaderMLLM(Settings(reader_hint=hint), "m", "reader_a", model=scripted('{"selected_option": "A", "confidence": 0.9}')[0])
    assert r.image_parts(st) == ([st.query_image_path, st.reference_image_path] + (["c"] if hint == "full" else []))
    r.image_parts = lambda s: []
    r.load()
    try:
        r._run(st, rec := {})
    finally:
        r.unload()
    assert all(("Visual expert" in c["prompt"]) == (hint == "full") for c in rec["calls"])


def test_hint_gated_by_source():
    s = Settings(reader_hint_sources=["VisA"])
    assert s.hint_on("VisA") and not s.hint_on("GoodsAD")
    assert Settings().hint_on("GoodsAD") and not Settings(reader_hint="none").hint_on("VisA")
    assert InspectionState.from_sample(Sample(image_id="i", image_path="i.png", source="VisA", templates=["r.png"],
                                              questions=[])).source == "VisA"


def test_reader_samples_majority_vote_for_iso_compute_baseline():
    st = mk_state()
    st.questions = st.questions[1:]
    model, calls = scripted('{"selected_option": "B", "confidence": 0.9}', '{"selected_option": "C", "confidence": 0.6}',
                            '{"selected_option": "C", "confidence": 0.8}')
    r = ReaderMLLM(Settings(reader_samples=3), "m", "reader_a", model=model)
    r.image_parts = lambda s: []
    r.load()
    try:
        st = r._run(st, rec := {})
    finally:
        r.unload()
    a = st.reader_a_responses[st.questions[0].id]
    assert a.selected_option == "C" and abs(a.confidence - 0.7) < 1e-9 and a.brief_rationale.startswith("vote 2/3")
    assert len(calls) == 3 and [c["sampling"]["seed"] for c in rec["calls"]] == [0, 1, 2]


def test_reader_accepts_schema_echo_of_internvl():
    st = mk_state()
    st.questions = st.questions[:1]
    echoed = ('{"properties": {"selected_option": "B", "confidence": 0.95, "brief_rationale": "identical"}, '
              '"required": ["selected_option", "confidence"], "additionalProperties": false}')
    model, calls = scripted(echoed)
    st, rec = run_reader(model, st)
    a = st.reader_a_responses[st.questions[0].id]
    assert a.selected_option == "B" and a.confidence == 0.95 and len(calls) == 1  # no retry needed
    from inspector.schemas.agent_outputs import ReaderAnswer
    assert ReaderAnswer.model_validate({"selected_option": "A", "confidence": 0.5, "properties": {"x": 1}}).selected_option == "A"


def test_reader_accepts_answer_and_rationale_keys_of_qwen_on_vllm():
    st = mk_state()
    st.questions = st.questions[:1]
    model, calls = scripted('{"answer": "B", "confidence": 0.95, "rationale": "hole"}')
    st, rec = run_reader(model, st)
    a = st.reader_a_responses[st.questions[0].id]
    assert (a.selected_option, a.brief_rationale, len(calls)) == ("B", "hole", 1)
    from inspector.schemas.agent_outputs import ReaderAnswer
    both = ReaderAnswer.model_validate({"selected_option": "A", "answer": "B", "confidence": 0.5})
    assert both.selected_option == "A"  # the schema key wins


def test_failed_reopen_keeps_the_first_answer():
    st = mk_state()
    q0, q1 = st.questions
    st.visual_evidence = expert(0.9, "top right")
    st.reader_a_responses = {q0.id: ans(q0.id, "B"), q1.id: None}
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    st = CriticVerifier(Settings())._run(st, {})
    r = ReaderMLLM(Settings(), "m", "reader_a_reopen", reopen=True, model=scripted("garbage")[0])
    r.image_parts = lambda s: []
    r.load()
    try:
        st = r._run(st, rec := {})
    finally:
        r.unload()
    assert st.reader_a_responses[q0.id].selected_option == "B" and st.reader_a_responses[q1.id] is None
    assert all("error" in c for c in rec["calls"])  # the failure stays visible in the log


def test_critic_without_tool_sends_no_tool_and_does_not_mention_it():
    seen = []

    def fn(messages, info):
        seen.append((info.function_tools, str(messages)))
        return ModelResponse(parts=[TextPart('{"is_valid": true, "conflict_detected": false}')])

    st = mk_state()
    c = CriticVerifier(Settings(critic_use_tool=False), model=FunctionModel(fn))
    c.image_parts = lambda s: []
    c.load()
    try:
        c._run(st, rec := {})
    finally:
        c.unload()
    assert seen[0][0] == [] and "run_consistency_rules" not in seen[0][1] and "llm" in rec


def test_reader_agent_gives_none_after_retries():
    model, _ = scripted("not json at all")
    st, rec = run_reader(model, mk_state())
    assert all(v is None for v in st.reader_a_responses.values())
    assert all("error" in c for c in rec["calls"])


# ---- critic ------------------------------------------------------------------
def test_critic_conflict_reopens_once():
    s = Settings()
    st = mk_state()
    st.visual_evidence = expert(0.9, "top right")
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "B"), q1.id: ans(q1.id, "A")}  # "No defect" + wrong cell
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    critic = CriticVerifier(s)
    st = critic._run(st, {})
    assert st.critic_verdict.conflict_detected and set(st.critic_verdict.reopen_questions) == {q0.id, q1.id}
    assert st.critic_verdict.reopen_by_reader == {"reader_a": [q0.id, q1.id], "reader_b": []}  # B was consistent
    assert st.reopen_count == 1
    st2 = critic._run(st.model_copy(deep=True), {})  # a second critic run must NOT grant another reopen
    assert st2.reopen_count == 1 and st2.critic_verdict.reopen_questions == []
    a = ReaderMLLM(s, "m", "reader_a_reopen", reopen=True, model=scripted("x")[0])
    b = ReaderMLLM(s, "m", "reader_b_reopen", reopen=True, model=scripted("x")[0])
    assert a.applies(st) and not b.applies(st)          # targeted: only reader A re-answers
    assert "reader_a:" in a._reopen_for_me(st)[1] and "reader_b:" not in a._reopen_for_me(st)[1]
    legacy = st.model_copy(deep=True)
    legacy.critic_verdict.reopen_by_reader = {}          # verdict cached before targeted reopen
    assert b.applies(legacy)


def test_critic_flags_missing_answer_as_invalid():
    st = mk_state()
    st.reader_a_responses = {q.id: None for q in st.questions}
    st.reader_b_responses = {q.id: ans(q.id, "A") for q in st.questions}
    st = CriticVerifier(Settings())._run(st, {})
    assert not st.critic_verdict.is_valid


def test_llm_critic_calls_rule_tool_rejects_unknown_ids_and_merges_with_rules():
    st = mk_state()
    st.visual_evidence = expert(0.9, "top right")
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "B"), q1.id: ans(q1.id, "B")}  # rules: q0 contradicts expert
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    bad = '{"is_valid": true, "conflict_detected": true, "reopen_questions": ["nope"], "critique_reason": "x"}'
    good = f'{{"is_valid": true, "conflict_detected": true, "reopen_questions": ["{q1.id}"], "critique_reason": "loc looks wrong"}}'
    model, calls = scripted(("tool", "run_consistency_rules"), bad, good)
    critic = CriticVerifier(Settings(json_retries=1), model=model)
    critic.image_parts = lambda s: []
    critic.load()
    try:
        st = critic._run(st, rec := {})
    finally:
        critic.unload()
    tool_out = [p.content for m in calls[1] for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
    assert "said no defect" in tool_out[0]                       # the rules reached the LLM via the tool
    assert set(st.critic_verdict.reopen_questions) == {q0.id, q1.id}  # rules ∪ LLM, unknown id retried away
    assert "reviewer: loc looks wrong" in st.critic_verdict.critique_reason


def test_critic_logs_its_tool_calls():
    from inspector.agents.mock import MockCritic
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    c = MockCritic(Settings())
    c.load()
    try:
        c._run(st, rec := {})
    finally:
        c.unload()
    assert rec["tool_calls"] == 1 and "llm" in rec


def test_llm_critic_schema_has_only_its_four_fields_and_prompt_has_rule_findings():
    from inspector.schemas.agent_outputs import CriticReview
    assert set(CriticReview.model_json_schema()["properties"]) == {"is_valid", "conflict_detected", "reopen_questions", "critique_reason"}
    st = mk_state()
    st.visual_evidence = expert(0.9, "top right")
    q0, q1 = st.questions
    st.reader_a_responses = st.reader_b_responses = {q0.id: ans(q0.id, "B"), q1.id: ans(q1.id, "B")}
    model, _ = scripted('{"is_valid": true, "conflict_detected": false, "reopen_questions": [], "critique_reason": ""}')
    c = CriticVerifier(Settings(), model=model)
    c.image_parts = lambda s: []
    c.load()
    try:
        c._run(st, rec := {})
    finally:
        c.unload()
    assert "Automatic checks:" in rec["prompt"] and "said no defect" in rec["prompt"] and "llm_error" not in rec


def test_llm_critic_failure_keeps_rules_verdict():
    st = mk_state()
    st.visual_evidence = expert(0.9, "top right")
    q0, q1 = st.questions
    st.reader_a_responses = st.reader_b_responses = {q0.id: ans(q0.id, "B"), q1.id: ans(q1.id, "B")}
    model, _ = scripted("garbage")
    critic = CriticVerifier(Settings(), model=model)
    critic.image_parts = lambda s: []
    critic.load()
    st = critic._run(st, rec := {})
    critic.unload()
    assert st.critic_verdict.reopen_questions == [q0.id] and "llm_error" in rec


def test_llm_critic_server_error_keeps_rules_verdict():
    """A 5xx from the server (Ollama: "token repeat limit reached") must not crash the pass."""
    from pydantic_ai.exceptions import ModelServerError

    def boom(messages, info):
        raise ModelServerError("gemma4:e2b", "prediction aborted, token repeat limit reached")

    st = mk_state()
    st.visual_evidence = expert(0.9, "top right")
    q0, q1 = st.questions
    st.reader_a_responses = st.reader_b_responses = {q0.id: ans(q0.id, "B"), q1.id: ans(q1.id, "B")}
    critic = CriticVerifier(Settings(), model=FunctionModel(boom))
    critic.image_parts = lambda s: []
    critic.load()
    st = critic._run(st, rec := {})
    critic.unload()
    assert st.critic_verdict.reopen_questions == [q0.id] and "ModelServerError" in rec["llm_error"]


def test_critic_extra_body_reaches_model_settings():
    off = CriticVerifier(Settings(), model=scripted("x")[0]).build(scripted("x")[0])
    assert "extra_body" not in (off.model_settings or {})
    s = Settings(critic_extra_body={"reasoning_effort": "none"})
    on = CriticVerifier(s, model=scripted("x")[0]).build(scripted("x")[0])
    assert on.model_settings["extra_body"] == {"reasoning_effort": "none"}


def test_judge_extra_body_reaches_model_settings():
    m = scripted("x")[0]
    assert "extra_body" not in (ConflictJudge(Settings(judge_model="j"), model=m).build(m).model_settings or {})
    on = ConflictJudge(Settings(judge_model="j", judge_extra_body={"reasoning_effort": "none"}), model=m).build(m)
    assert on.model_settings["extra_body"] == {"reasoning_effort": "none"}


def test_cell_in_text_longest_match():
    assert cell_in_text("top-center area") == "top center" and cell_in_text("in the center") == "center"


# ---- aggregator --------------------------------------------------------------
def test_aggregator_weighted_vote_and_tiebreak():
    s = Settings(weights={"reader_a": {"defect_localization": 2.0}})
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "A", 0.6), q1.id: ans(q1.id, "A")}
    st.reader_b_responses = {q0.id: ans(q0.id, "B", 0.9), q1.id: ans(q1.id, "B")}
    st = Aggregator(s)._run(st, {})
    assert st.final_answers[q1.id].selected_option == "A"   # weight 2 beats 1
    assert st.final_answers[q0.id].selected_option == "B"   # tie 1-1 -> higher confidence


def test_aggregator_expert_votes_on_discrimination_and_none_if_no_votes():
    st = mk_state()
    q0, q1 = st.questions
    st.visual_evidence = expert(0.9)
    st.reader_a_responses = {q0.id: ans(q0.id, "B"), q1.id: None}
    st.reader_b_responses = {q0.id: None, q1.id: None}
    st = Aggregator(Settings(expert_weight_discrimination=2.0))._run(st, {})
    assert st.final_answers[q0.id].selected_option == "A"  # expert "Yes" (weight 2) outvotes reader "No" (1)
    assert st.final_answers[q1.id].selected_option is None


def test_aggregator_expert_vote_per_source():
    st = mk_state()
    q0, _ = st.questions
    st.visual_evidence = expert(0.6)
    st.reader_a_responses = {q0.id: ans(q0.id, "B")}
    st.reader_b_responses = {}
    s = Settings(expert_weight_discrimination=2.0)
    assert Aggregator(s)._run(st.model_copy(deep=True), {}).final_answers[q0.id].selected_option == "A"
    st.source = "GoodsAD"
    off = Settings(expert_weight_discrimination=2.0, expert_weight_by_source={"GoodsAD": 0})
    assert Aggregator(off)._run(st.model_copy(deep=True), {}).final_answers[q0.id].selected_option == "B"
    hi = Settings(expert_weight_discrimination=2.0, expert_threshold_by_source={"GoodsAD": 0.7})
    out = Aggregator(hi)._run(st.model_copy(deep=True), {}).final_answers[q0.id]
    assert out.selected_option == "B" and out.votes["B"] == 3.0  # 0.6 < 0.7: expert says "no defect" too


# ---- metrics -----------------------------------------------------------------
def rows(spec):
    return [{"image_id": i, "question_id": f"{i}#0", "subtask": SubTask.ANOMALY_DISCRIMINATION,
             "correct": c, "is_defective": d, "source": "x"} for i, (c, d) in enumerate(spec)]


def test_aggregator_third_reader_breaks_tie_and_can_be_left_out():
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "A", 0.6), q1.id: ans(q1.id, "C")}
    st.reader_b_responses = {q0.id: ans(q0.id, "B", 0.9), q1.id: ans(q1.id, "D")}
    st.reader_c_responses = {q0.id: ans(q0.id, "A", 0.5), q1.id: ans(q1.id, "D")}
    out = Aggregator(Settings())._run(st.model_copy(deep=True), {})
    assert out.final_answers[q0.id].selected_option == "A"   # 2 votes vs 1 (B had the higher confidence)
    assert out.final_answers[q1.id].selected_option == "D"
    ab = Aggregator(Settings(aggregate_readers=["reader_a", "reader_b"]))._run(st.model_copy(deep=True), {})
    assert ab.final_answers[q0.id].selected_option == "B"    # without C: tie -> confidence


@pytest.mark.parametrize("mode,expected", [("override", "C"), ("vote", "C"), ("off", "B")])
def test_aggregator_judge_modes(mode, expected):
    st = mk_state()
    q1 = st.questions[1]
    st.reader_a_responses = {q1.id: ans(q1.id, "B", 0.9)}
    st.reader_b_responses = {q1.id: ans(q1.id, "C", 0.5)}
    st.judge_answers = {q1.id: ans(q1.id, "C", 0.7)}
    st = Aggregator(Settings(judge_mode=mode))._run(st, {})
    assert st.final_answers[q1.id].selected_option == expected


def test_conflicts_only_where_readers_disagree():
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "C")}
    st.reader_c_responses = {q0.id: None, q1.id: None}
    got = conflicts(st, ["reader_a", "reader_b", "reader_c"])
    assert [q.id for q, _ in got] == [q1.id] and {a.selected_option for a in got[0][1]} == {"B", "C"}
    st.reader_b_responses[q1.id] = None           # one valid answer left: nothing to choose
    assert conflicts(st, ["reader_a", "reader_b"]) == []


def test_judge_must_pick_a_candidate_and_sees_only_candidates():
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B")}
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "C")}
    # "A" is offered by the question but proposed by no reader -> ModelRetry -> "C" kept
    model, calls = scripted('{"selected_option": "A", "confidence": 0.8}', '{"selected_option": "C", "confidence": 0.8}')
    j = ConflictJudge(Settings(judge_model="m"), model=model)
    j.image_parts = lambda s: []
    assert j.applies(st)
    j.load()
    try:
        st = j._run(st, rec := {})
    finally:
        j.unload()
    assert list(st.judge_answers) == [q1.id] and st.judge_answers[q1.id].selected_option == "C"
    prompt = rec["calls"][0]["prompt"]
    assert "B. top right" in prompt and "C. center" in prompt and "A. top left" not in prompt
    assert len(calls) == 2


def test_llm_aggregator_decides_only_contested_questions_and_falls_back():
    st = mk_state()
    q0, q1 = st.questions
    st.reader_a_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "B", 0.9)}
    st.reader_b_responses = {q0.id: ans(q0.id, "A"), q1.id: ans(q1.id, "C", 0.5)}
    st.judge_answers = {q1.id: ans(q1.id, "C", 0.7)}
    # "A" is offered by the question but is no candidate -> ModelRetry -> "C"
    model, calls = scripted('{"selected_option": "A", "confidence": 0.8}', '{"selected_option": "C", "confidence": 0.8}')
    agg = LLMAggregator(Settings(aggregator_model="m"), model=model)
    assert [q.id for q, _, _ in agg.contested(st)] == [q1.id]   # q0 is unanimous: no LLM call
    agg.load()
    try:
        out = agg._run(st.model_copy(deep=True), rec := {})
    finally:
        agg.unload()
    assert out.final_answers[q0.id].selected_option == "A" and out.final_answers[q1.id].selected_option == "C"
    assert len(calls) == 2 and rec["llm_decided"] == [q1.id]
    prompt = rec["calls"][0]["prompt"]
    assert "Senior inspector" in prompt and "B. top right" in prompt and "A. top left" not in prompt
    # model failure: the weighted vote stands (judge_mode "off" here, B has the higher confidence on a tie)
    bad, _ = scripted("not json", "not json", "not json", "not json")
    agg = LLMAggregator(Settings(aggregator_model="m", judge_mode="off"), model=bad)
    agg.load()
    try:
        out = agg._run(st.model_copy(deep=True), {})
    finally:
        agg.unload()
    assert out.final_answers[q1.id].selected_option == "B"


def test_optional_passes_built_only_when_configured(tmp_path):
    names = lambda s: [st.name for st in build_steps(s, tmp_path, mock=True)]  # noqa: E731
    assert "3b_reader_c" not in names(Settings()) and "6b_judge" not in names(Settings())
    full = names(Settings(reader_c_model="c", judge_model="j"))
    assert full.index("3b_reader_c") < full.index("4_critic") < full.index("6b_judge") < full.index("7_aggregator")


def test_full_mock_pipeline_with_third_reader_and_judge(tmp_path):
    s = Settings(reader_c_model="c", judge_model="j")
    samples = [mk_sample(i) for i in range(4)]
    states = [InspectionState.from_sample(x) for x in samples]
    out = PassRunner(tmp_path, Telemetry(tmp_path / "t.jsonl")).run(build_steps(s, tmp_path, mock=True), states)
    assert all(len(st.final_answers) == 2 and len(st.reader_c_responses) == 2 for st in out)
    assert (tmp_path / "6b_judge").is_dir()


def test_full_mock_pipeline_with_llm_aggregator(tmp_path):
    s = Settings(reader_c_model="c", judge_model="j", aggregator_model="a")
    states = [InspectionState.from_sample(mk_sample(i)) for i in range(4)]
    out = PassRunner(tmp_path, Telemetry(tmp_path / "t.jsonl")).run(build_steps(s, tmp_path, mock=True), states)
    assert all(len(st.final_answers) == 2 for st in out) and (tmp_path / "7_aggregator").is_dir()


def test_balanced_accuracy_and_fpr():
    r = rows([(True, False), (True, False), (True, False), (False, True)])  # 3/3 normal, 0/1 defective
    assert metrics.balanced_accuracy(r) == pytest.approx(0.5)
    assert metrics.fpr_and_miss(r) == (0.0, 1.0)


def test_mcnemar_and_bootstrap():
    a = rows([(True, True)] * 20 + [(False, True)] * 0)
    b = rows([(False, True)] * 20)
    n_a, n_b, p = metrics.mcnemar_exact(a, b)
    assert (n_a, n_b) == (20, 0) and p < 1e-4
    assert metrics.mcnemar_exact(a, a)[2] == 1.0
    lo, hi = metrics.bootstrap_ci(rows([(True, True), (False, False)] * 10), n_boot=100)
    assert 0 <= lo <= hi <= 1


def test_split_is_stratified_and_deterministic():
    samples = [mk_sample(i, defective=i % 2 == 0) for i in range(20)]
    d1, t1 = make_split(samples, 0.1, seed=0)
    d2, _ = make_split(samples, 0.1, seed=0)
    assert [s.image_id for s in d1] == [s.image_id for s in d2]
    assert len(d1) + len(t1) == 20 and not {s.image_id for s in d1} & {s.image_id for s in t1}
    assert {s.is_defective for s in d1} == {True, False}


# ---- pass runner -------------------------------------------------------------
class Spy(MockExpert):
    events: list = []

    def load(self):
        Spy.events.append("load")

    def unload(self):
        Spy.events.append("unload")
        super().unload()


def test_sample_per_source_is_balanced_and_deterministic():
    pool = [Sample(image_id=f"{src}/c/{d}/{i}", image_path="x.png", source=src, is_defective=d,
                   templates=["r.png"], questions=mk_questions(f"{src}/c/{d}/{i}"))
            for src in ("VisA", "GoodsAD") for d in (False, True) for i in range(5)]
    a = sample_per_source(pool, 4, seed=0)
    assert len(a) == 8 and a == sample_per_source(pool, 4, seed=0)
    assert {(s.source, s.is_defective) for s in a} == {(src, d) for src in ("VisA", "GoodsAD") for d in (False, True)}


def test_stratified_subsample_exact_size_proportional_and_deterministic():
    pool = [Sample(image_id=f"{src}/c/{d}/{i}", image_path="x.png", source=src, is_defective=d,
                   templates=["r.png"], questions=mk_questions(f"{src}/c/{d}/{i}"))
            for src, size in (("VisA", 60), ("GoodsAD", 30)) for d in (False, True) for i in range(size)]
    a = stratified_subsample(pool, 45, seed=0)
    assert len(a) == 45 and len({s.image_id for s in a}) == 45
    assert sum(s.source == "VisA" for s in a) == 30          # 2/3 of the pool -> 2/3 of the sample
    assert [s.image_id for s in a] == [s.image_id for s in stratified_subsample(pool, 45, seed=0)]
    assert len(stratified_subsample(pool, 10_000, seed=0)) == len(pool)


def test_pass_runner_unloads_each_pass_and_resumes_from_cache(tmp_path):
    Spy.events = []
    tel = Telemetry(tmp_path / "t.jsonl")
    steps = [Step("1_expert", Spy), Step("2_expert_again", Spy)]
    runner = PassRunner(tmp_path, tel)
    states = [mk_state(i) for i in range(3)]
    out = runner.run(steps, states)
    assert Spy.events == ["load", "unload", "load", "unload"]  # one model at a time
    assert all(s.visual_evidence for s in out)
    Spy.events = []
    runner.run(steps, [mk_state(i) for i in range(3)])  # all cached -> nothing loaded
    assert Spy.events == []
    assert any("latency_ms" in json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines())


def test_pass_runner_caches_each_image_before_a_crash(tmp_path):
    class Boom(MockExpert):
        n = 0

        def _run(self, state, rec):
            Boom.n += 1
            if Boom.n == 3:
                raise RuntimeError("crash")
            return super()._run(state, rec)

    states = [mk_state(i) for i in range(4)]
    with pytest.raises(RuntimeError):
        PassRunner(tmp_path, Telemetry(tmp_path / "t.jsonl")).run([Step("1_expert", Boom)], states)
    assert len(list((tmp_path / "1_expert").glob("*.json"))) == 2  # the two finished images survived
    out = PassRunner(tmp_path, Telemetry(tmp_path / "t.jsonl")).run([Step("1_expert", MockExpert)], states)
    assert all(s.visual_evidence for s in out)


def test_full_mock_pipeline_end_to_end(tmp_path):
    s = Settings()
    tel = Telemetry(tmp_path / "t.jsonl")
    samples = [mk_sample(i) for i in range(4)]
    states = [InspectionState.from_sample(x) for x in samples]
    out = PassRunner(tmp_path, tel).run(build_steps(s, tmp_path, mock=True), states)
    assert all(len(st.final_answers) == 2 for st in out)
    assert all(st.reopen_count <= s.max_reopen for st in out)
    r = metrics.build_rows(samples, out)
    assert 0 <= metrics.mean_accuracy(r) <= 1


# ---- visual expert (pure functions; model loading needs weights, not tested here) ----
def test_expert_anomaly_map_and_grid():
    import numpy as np
    from inspector.agents.visual_expert import anomaly_map, grid_cell
    r = np.eye(4, dtype=np.float32)
    q = np.vstack([r[:3], [[1, 1, 1, 1]]]).astype(np.float32)  # 3 patches match reference, last one differs
    d = anomaly_map(q, r)
    assert d[:3].max() < 1e-6 and d[3] > 0.4
    assert grid_cell(0, 0, 90, 90) == "top left" and grid_cell(89, 89, 90, 90) == "bottom right" and grid_cell(45, 45, 90, 90) == "center"


def test_reader_fails_fast_without_llm_server():
    r = ReaderMLLM(Settings(_env_file=None, llm_provider="vllm", llm_base_url="http://127.0.0.1:9/v1", vllm_autostart=False), "Qwen/Qwen3-VL-8B-Instruct", "reader_a")
    with pytest.raises(RuntimeError, match="No LLM server"):
        r.load()


def test_remote_role_uses_its_own_endpoint_and_no_local_server(monkeypatch):
    import inspector.agents.llm as llm

    monkeypatch.setattr(llm.VLLMServer, "start", lambda self: pytest.fail("local server must not start"))
    s = Settings(_env_file=None, backup_enabled=False, remote={"reader_a": {"base_url": "https://api.x.ai/v1", "api_key": "k", "model": "grok-x"}})
    r = ReaderMLLM(s, "Qwen/Qwen3-VL-8B-Instruct", "reader_a_reopen", reopen=True)
    r.load()
    try:
        assert r.agent.model.model_name == "grok-x" and "api.x.ai" in str(r.agent.model.client.base_url)
    finally:
        r.unload()


def test_grok_fills_remote_for_three_roles():
    s = Settings(_env_file=None, use_grok=True, grok_api_key="k")
    assert set(s.remote) == {"reader_a", "reader_b", "critic"}
    assert all("api.x.ai" in ep["base_url"] and ep["api_key"] == "k" for ep in s.remote.values())
    assert "api.x.ai" in make_model(s, "x", s.remote["critic"]).client.base_url.host


def test_grok_without_key_is_an_error(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROK_API_KEY"):
        Settings(_env_file=None, use_grok=True)


def test_router_is_default_and_starts_no_server(monkeypatch):
    import inspector.agents.llm as llm

    monkeypatch.setattr(llm.VLLMServer, "start", lambda self: pytest.fail("router must not start a server"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    s = Settings(_env_file=None)  # ignore the local .env: this checks the built-in defaults
    r = ReaderMLLM(s, s.reader_a_model, "reader_a")
    r.load()
    try:
        assert "router.huggingface.co" in str(r.agent.model.client.base_url) and r.agent.model.client.api_key == "hf_test"
    finally:
        r.unload()


def test_remote_role_gets_a_backup_endpoint(monkeypatch):
    from pydantic_ai.models.fallback import FallbackModel
    s = Settings(_env_file=None, llm_api_key="gem-key", use_grok=True, grok_api_key="groq-key")
    assert isinstance(make_role_model(s, "gemini-x", s.remote["reader_a"]), FallbackModel)
    assert not isinstance(make_role_model(s, "gemini-x"), FallbackModel)            # no remote: nothing to back up
    off = Settings(_env_file=None, llm_api_key="gem-key", use_grok=True, grok_api_key="groq-key", backup_enabled=False)
    assert not isinstance(make_role_model(off, "gemini-x", off.remote["reader_a"]), FallbackModel)
    nokey = Settings(_env_file=None, use_grok=True, grok_api_key="groq-key")        # llm_api_key stays "EMPTY"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not isinstance(make_role_model(nokey, "gemini-x", nokey.remote["reader_a"]), FallbackModel)


def test_backup_answers_when_primary_hits_quota():
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.fallback import FallbackModel

    def quota(messages, info):
        raise ModelHTTPError(429, "groq", {"error": "rate limit"})

    def ok(messages, info):
        return ModelResponse(parts=[TextPart("from backup")])

    agent = Agent(FallbackModel(FunctionModel(quota), FunctionModel(ok)))
    assert agent.run_sync("hi").output == "from backup"


def test_load_mmad_available_under_keeps_only_existing_images(tmp_path):
    entry = lambda img, ref: {"random_templates": [ref], "conversation": [
        {"Question": "q?", "Options": {"A": "a", "B": "b"}, "Answer": "A", "type": "Anomaly Discrimination"}]}
    data = {"S/c/test/good/1.png": entry(0, "S/c/train/good/0.png"), "S/c/test/good/2.png": entry(0, "S/c/train/good/0.png")}
    (tmp_path / "mmad.json").write_text(json.dumps(data))
    for p in ("S/c/test/good/1.png", "S/c/train/good/0.png"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"x")
    assert len(load_mmad(tmp_path / "mmad.json")) == 2
    kept = load_mmad(tmp_path / "mmad.json", available_under=tmp_path)
    assert [x.image_path for x in kept] == ["S/c/test/good/1.png"]


def test_multiple_keys_build_a_failover_chain():
    s = Settings(_env_file=None, use_grok=True, grok_api_key="g1", grok_api_keys="g2, g3",
                 llm_api_key="m1", llm_api_keys="m2 m3")
    fm = make_role_model(s, "gemini-x", s.remote["reader_a"])
    keys = [m.client.api_key for m in fm.models]
    assert keys == ["g1", "g2", "g3", "m1", "m2", "m3"]
    only_gemini = Settings(_env_file=None, llm_api_key="m1", llm_api_keys="m2")      # no hosted endpoint: rotate Gemini keys
    assert [m.client.api_key for m in make_role_model(only_gemini, "gemini-x").models] == ["m1", "m2"]
    assert not hasattr(make_role_model(Settings(_env_file=None, llm_api_key="m1"), "gemini-x"), "models")


def test_pass_runner_prints_progress(tmp_path, capsys):
    states = [mk_state(i) for i in range(12)]
    steps = [Step("1_expert", MockExpert)]
    PassRunner(tmp_path, Telemetry(tmp_path / "t.jsonl", run="r")).run(steps, states)
    out = capsys.readouterr().out
    assert "1_expert 10/12" in out and "1_expert 12/12" in out and "1_expert 11/12" not in out


class _FakeProc:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def _vllm(monkeypatch, codes):
    """VLLMServer whose successive `vllm serve` processes exit with `codes` (None = stays up and serves)."""
    from inspector.agents import llm
    procs, waits = [], []

    def popen(cmd):
        procs.append(_FakeProc(codes[len(procs)]))
        return procs[-1]

    monkeypatch.setattr(llm.subprocess, "Popen", popen)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm, "served_models", lambda s: ["m"] if procs[-1].returncode is None else None)
    return llm.VLLMServer(Settings(vllm_autostart=True), "m"), procs, waits


def test_vllm_server_retries_after_a_failed_start(monkeypatch):
    server, procs, waits = _vllm(monkeypatch, [1, None])
    server.start()
    assert len(procs) == 2 and server.proc is procs[1] and 30 in waits  # waits for GPU memory before retrying


def test_vllm_server_gives_up_after_the_last_retry(monkeypatch):
    server, procs, _ = _vllm(monkeypatch, [1, 1, 1, None])
    with pytest.raises(RuntimeError, match="exited with code 1"):
        server.start()
    assert len(procs) == 3 and server.proc is None
