"""GPU-free tests: schemas, strict 1-shot, critic/reopen guard, aggregator, metrics, pass runner."""
import json

import pytest
from pydantic import ValidationError

from inspector.agents.aggregator import Aggregator
from inspector.agents.critic_verifier import CriticVerifier, cell_in_text
from inspector.agents.mock import MockExpert, MockReader
from inspector.agents.reader_mllm import parse_reader_json
from inspector.config.settings import Settings
from inspector.orchestrator.graph import Step, build_steps
from inspector.orchestrator.pass_runner import PassRunner
from inspector.schemas.agent_outputs import ReaderOutput, VisualExpertOutput
from inspector.schemas.mmad import Question, Sample, SubTask, load_mmad
from inspector.schemas.state import InspectionState
from inspector.utils import metrics
from inspector.utils.splits import make_split
from inspector.utils.telemetry import Telemetry

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


def test_parse_reader_json_tolerant():
    q = mk_questions()[0]
    assert parse_reader_json('Sure! {"selected_option": "b", "confidence": 0.7} done', q).selected_option == "B"
    assert parse_reader_json('{"selected_option": "C", "confidence": 0.7}', q) is None  # not an offered option
    assert parse_reader_json("no json", q) is None


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
    assert st.reopen_count == 1
    st = critic._run(st, {})  # a second critic run must NOT grant another reopen
    assert st.reopen_count == 1 and st.critic_verdict.reopen_questions == []


def test_critic_flags_missing_answer_as_invalid():
    st = mk_state()
    st.reader_a_responses = {q.id: None for q in st.questions}
    st.reader_b_responses = {q.id: ans(q.id, "A") for q in st.questions}
    st = CriticVerifier(Settings())._run(st, {})
    assert not st.critic_verdict.is_valid


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


# ---- metrics -----------------------------------------------------------------
def rows(spec):
    return [{"image_id": i, "question_id": f"{i}#0", "subtask": SubTask.ANOMALY_DISCRIMINATION,
             "correct": c, "is_defective": d, "source": "x"} for i, (c, d) in enumerate(spec)]


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
