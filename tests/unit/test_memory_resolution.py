import pytest

import continuator_memory.memory.context as context
from continuator_memory.evaluation.dataset import parse_dataset
from continuator_memory.evaluation.harness import run_evaluation
from continuator_memory.memory.context import build_memory_prompt, render_memory_context
from continuator_memory.memory.extraction import MemoryExtractor
from continuator_memory.memory.pipeline import MemoryPipeline
from continuator_memory.store import Store
from tests.helpers import ScriptedJSONClient
from tests.unit.test_question_understanding import MH05, MH06, _team_offsite

MH05_RESOLUTION = (
    "\n\nRESOLUTION:\n"
    '- "the person who handles the offsite budget" is Mei (handles: budget)\n'
    '- requested "city" of Mei: origin: Taipei [CURRENT]'
)
MH06_RESOLUTION = (
    "\n\nRESOLUTION:\n"
    '- "the team member whose birthday falls during the offsite" is Omar '
    "(birthday: September 8th)\n"
    '- requested "city" of Omar: origin: Cairo [CURRENT]'
)
NO_PIVOT_QUESTIONS = [
    "Where is Omar from?",
    "What instrument does Mei play?",
    "Who is responsible for booking the train tickets?",
    "What is Omar's last name?",
    "Which team member cannot go on boat trips, and what is their role?",
]


class RecordingGenerator:
    model_name = "recording"

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "answer"


@pytest.fixture
def offsite(embedder):
    conv, script = _team_offsite()
    client = ScriptedJSONClient(script)
    with Store(":memory:") as store:
        p = MemoryPipeline(store, embedder, RecordingGenerator(), MemoryExtractor(client),
                           window_size=4, overlap=2)
        p.ingest(conv)
        yield p, client


def _phase35(monkeypatch, fn, *args):
    with monkeypatch.context() as m:
        m.setattr(context, "_resolution_lines", lambda retrieval: [])
        return fn(*args)


@pytest.mark.parametrize("question, resolution", [(MH05, MH05_RESOLUTION),
                                                  (MH06, MH06_RESOLUTION)])
def test_constraint_pivot_renders_resolution(offsite, question, resolution):
    p, _ = offsite
    text = render_memory_context(p.retrieve_memory(question, 5, "team-offsite"))
    assert text.endswith(resolution)
    assert text.count("RESOLUTION:") == 1


@pytest.mark.parametrize("question", [MH05, MH06])
def test_memory_lines_and_source_evidence_unchanged(offsite, monkeypatch, question):
    p, _ = offsite
    r = p.retrieve_memory(question, 5, "team-offsite")
    new_memory, old_memory = render_memory_context(r), _phase35(monkeypatch,
                                                                render_memory_context, r)
    assert new_memory.startswith(old_memory + "\n\nRESOLUTION:\n")
    new_prompt = build_memory_prompt(question, r)
    old_prompt = _phase35(monkeypatch, build_memory_prompt, question, r)
    marker = "\n\nSOURCE EVIDENCE (conversation excerpts):"
    assert new_prompt.split(marker)[1] == old_prompt.split(marker)[1]
    assert new_prompt.index("RESOLUTION:") < new_prompt.index(marker)
    assert new_prompt.replace(new_memory, old_memory, 1) == old_prompt


def test_prompt_is_byte_identical_without_pivots(offsite, monkeypatch):
    p, _ = offsite
    for q in NO_PIVOT_QUESTIONS:
        r = p.retrieve_memory(q, 5, "team-offsite")
        assert r.understanding["pivots"] == []
        prompt = build_memory_prompt(q, r)
        assert prompt == _phase35(monkeypatch, build_memory_prompt, q, r)
        assert "RESOLUTION" not in prompt


def test_resolution_requires_no_llm_call(offsite):
    p, client = offsite
    extraction_calls = len(client.prompts)
    answer = p.ask(MH05, 5, "team-offsite")
    assert len(p.generator.calls) == 1
    assert len(client.prompts) == extraction_calls
    assert MH05_RESOLUTION in answer.prompt
    assert p.generator.calls[0][1] == answer.prompt


def test_retrieval_metrics_unchanged(embedder, monkeypatch):
    conv, script = _team_offsite()
    dataset = parse_dataset({
        "dataset_id": "resolution-smoke",
        "conversations": [{"conversation_id": conv.conversation_id, "messages": [
            {"id": m.message_id, "role": m.role, "content": m.content} for m in conv.messages]}],
        "questions": [
            {"id": "q5", "category": "multi_hop", "conversation_id": "team-offsite",
             "question": MH05, "expected_answer": "Taipei",
             "expected_source_message_ids": ["m07", "m13"]},
            {"id": "q6", "category": "multi_hop", "conversation_id": "team-offsite",
             "question": MH06, "expected_answer": "Cairo",
             "expected_source_message_ids": ["m05", "m15"]},
        ],
    })

    def run():
        with Store(":memory:") as store:
            p = MemoryPipeline(store, embedder, RecordingGenerator(),
                               MemoryExtractor(ScriptedJSONClient(script)),
                               window_size=4, overlap=2)
            res = run_evaluation(dataset, p, generation_top_k=5)
        return res, [c[1] for c in p.generator.calls]

    new, new_prompts = run()
    old, old_prompts = _phase35(monkeypatch, run)
    assert new["aggregate"]["retrieval"] == old["aggregate"]["retrieval"]
    for a, b in zip(new["questions"], old["questions"]):
        timing = {"latency_ms"}
        assert ({k: v for k, v in a["retrieval"].items() if k not in timing}
                == {k: v for k, v in b["retrieval"].items() if k not in timing})
        assert a["answer"]["context_chunk_ids"] == b["answer"]["context_chunk_ids"]
        assert a["answer"]["memory"] == b["answer"]["memory"]
    assert all("RESOLUTION:" in x for x in new_prompts)
    assert not any("RESOLUTION:" in x for x in old_prompts)
