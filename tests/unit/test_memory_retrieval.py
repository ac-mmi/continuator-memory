import pytest

from continuator_memory.evaluation.dataset import parse_dataset
from continuator_memory.evaluation.harness import run_evaluation
from continuator_memory.memory.context import (
    MEMORY_SYSTEM_PROMPT,
    build_memory_prompt,
    render_memory_context,
)
from continuator_memory.memory.extraction import MemoryExtractor
from continuator_memory.memory.pipeline import MemoryPipeline
from continuator_memory.models import parse_conversation
from continuator_memory.pipeline import Pipeline
from continuator_memory.store import Store
from tests.helpers import HashingEmbedder, ScriptedJSONClient, doc, mem, rel

TEXTS = [
    "Project Nova launches on June 3.",
    "Priya handles the budget for Nova.",
    "Nova's launch moved to June 17.",
    "The office has a nice view of the harbour.",
    "Priya is from Oslo.",
    "Nova's launch is now July 1.",
    "The weather was rainy today.",
    "Omar enjoys hiking on weekends.",
]
SCRIPT = {
    TEXTS[0]: doc([mem("Project Nova", "launch_date", "June 3")]),
    TEXTS[1]: doc(relationships=[rel("Priya", "handles", "budget")]),
    TEXTS[2]: doc([mem("Nova", "launch_date", "June 17", "update", "June 3")]),
    TEXTS[4]: doc([mem("Priya", "from", "Oslo")]),
    TEXTS[5]: doc([mem("Nova", "launch_date", "July 1", "update")]),
    TEXTS[7]: doc([mem("Omar", "hobby", "hiking")]),
}
CONVERSATION = parse_conversation(
    {"conversation_id": "c1",
     "messages": [{"id": f"m{i + 1}", "role": "user", "content": t} for i, t in enumerate(TEXTS)]}
)


class RecordingGenerator:
    model_name = "recording"

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "answer"


@pytest.fixture
def pipeline(store, embedder):
    p = MemoryPipeline(store, embedder, RecordingGenerator(),
                       MemoryExtractor(ScriptedJSONClient(SCRIPT)), window_size=2, overlap=0)
    p.ingest(CONVERSATION)
    return p


def _slot(retrieval, entity, predicate):
    group = next(g for g in retrieval.groups if g.entity.name == entity)
    return next(s for s in group.slots if s[0].predicate == predicate)


def test_current_value_beats_superseded_for_current_state_query(pipeline):
    r = pipeline.retrieve_memory("When does Project Nova launch?", k=3, conversation_id="c1")
    assert r.analysis.temporal_scope == "current"
    assert [m.value for m in _slot(r, "Project Nova", "launch_date")] == ["July 1", "June 17",
                                                                           "June 3"]
    text = render_memory_context(r)
    assert text.index("July 1 [CURRENT") < text.index("June 17 [SUPERSEDED by July 1")
    assert "June 3 [SUPERSEDED by June 17" in text
    scores = pipeline.retriever._score(
        r.analysis, pipeline.base.retrieve("When does Project Nova launch?", 10, "c1"),
        pipeline.memory.memories("c1"), [],
    )
    by_value = {m.value: scores[m.memory_id] for m in pipeline.memory.memories("c1")}
    assert by_value["July 1"] > by_value["June 17"]
    assert by_value["July 1"] > by_value["June 3"]


def test_historical_value_remains_retrievable(pipeline):
    q = "What was the original launch date of Project Nova?"
    r = pipeline.retrieve_memory(q, k=3, conversation_id="c1")
    assert r.analysis.temporal_scope == "historical"
    values = {pipeline.memory.get_memory(mid).value for mid in r.trace()["memory_ids"]}
    assert {"June 3", "June 17", "July 1"} <= values
    assert ["c1", "m1"] in r.trace()["source_messages"]


def test_multi_hop_relationship_retrieval_combines_two_memories(pipeline):
    r = pipeline.retrieve_memory(
        "Which city is the person who handles the budget from?", k=3, conversation_id="c1"
    )
    assert [m.value for m in _slot(r, "Priya", "from")] == ["Oslo"]
    assert [(x.subject, x.predicate, x.object) for x in r.relationships] == [
        ("Priya", "handles", "budget")
    ]
    sources = r.trace()["source_messages"]
    assert ["c1", "m2"] in sources and ["c1", "m5"] in sources
    text = render_memory_context(r)
    assert "Priya -> handles -> budget [CURRENT]" in text and "from: Oslo [CURRENT]" in text


def test_unrelated_question_gets_no_memory(pipeline):
    r = pipeline.retrieve_memory("What is the capital of Peru?", k=3, conversation_id="c1")
    assert r.groups == [] and r.relationships == []
    assert "(no stored facts match this question)" in render_memory_context(r)
    assert len(r.chunks) == 3


def test_phase2_semantic_retrieval_unchanged(pipeline, embedder):
    fresh = Store(":memory:")
    baseline = Pipeline(fresh, embedder, window_size=2, overlap=0)
    baseline.ingest(CONVERSATION)
    q = "When does Project Nova launch?"
    assert pipeline.base.retrieve(q, 4, "c1") == baseline.retrieve(q, 4, "c1")
    fresh.close()


def test_memory_pipeline_without_memories_returns_semantic_ranking(store, embedder):
    p = MemoryPipeline(store, embedder, None, MemoryExtractor(ScriptedJSONClient({})),
                       window_size=2, overlap=0)
    p.ingest(CONVERSATION)
    q = "Priya budget"
    assert p.retrieve(q, 3, "c1") == p.base.retrieve(q, 3, "c1")


def test_fused_chunks_include_current_state_evidence(pipeline):
    chunks = pipeline.retrieve("When does Project Nova launch?", 3, "c1")
    assert len(chunks) == 3 and [c.rank for c in chunks] == [1, 2, 3]
    assert any("m6" in c.source_message_ids for c in chunks)


def test_ask_uses_memory_prompt(pipeline):
    answer = pipeline.ask("When does Project Nova launch?", k=2, conversation_id="c1")
    ((system, prompt),) = pipeline.generator.calls
    assert system == MEMORY_SYSTEM_PROMPT
    assert prompt == answer.prompt
    assert prompt.startswith("MEMORY:\nProject Nova\n- launch_date: July 1 [CURRENT")
    assert "SOURCE EVIDENCE (conversation excerpts):" in prompt
    assert prompt.endswith("Question: When does Project Nova launch?\nAnswer:")
    assert answer.memory_trace["n_superseded"] == 2
    assert len(answer.retrieved) == 2
    r = pipeline.retrieve_memory("When does Project Nova launch?", 2, "c1")
    assert build_memory_prompt("When does Project Nova launch?", r) == prompt


def test_harness_records_memory_diagnostics(embedder):
    dataset = parse_dataset({
        "dataset_id": "memory-smoke",
        "conversations": [{"conversation_id": "c1", "messages": [
            {"id": f"m{i + 1}", "role": "user", "content": t} for i, t in enumerate(TEXTS)]}],
        "questions": [
            {"id": "q1", "category": "temporal", "conversation_id": "c1",
             "question": "When does Project Nova launch?", "expected_answer": "July 1",
             "stale_answers": ["June 17"], "expected_source_message_ids": ["m6"]},
            {"id": "q2", "category": "multi_hop", "conversation_id": "c1",
             "question": "Which city is the person who handles the budget from?",
             "expected_answer": "Oslo", "expected_source_message_ids": ["m2", "m5"]},
        ],
    })

    class Scripted:
        model_name = "scripted"

        def generate(self, system, prompt):
            return "July 1." if "launch" in prompt.split("Question:")[-1] else "Oslo."

    with Store(":memory:") as store:
        p = MemoryPipeline(store, embedder, Scripted(), MemoryExtractor(ScriptedJSONClient(SCRIPT)),
                           window_size=2, overlap=0)
        res = run_evaluation(dataset, p, generation_top_k=3)
    assert res["aggregate"]["answer"]["overall"]["accuracy"] == 1.0
    mem_run = res["memory"]
    assert mem_run["memories"] == 5 and mem_run["memories_superseded"] == 2
    assert mem_run["consolidation_actions"]["memory"] == {"CREATE": 3, "SUPERSEDE": 2}
    assert mem_run["extraction_status"] == {"ok": 8}
    diag = res["aggregate"]["diagnostics"]
    assert diag["current_state"] == {"n": 1, "accuracy": 1.0, "clean_accuracy": 1.0}
    assert diag["memory"]["multi_source_coverage"] == 1.0
    assert res["questions"][1]["answer"]["memory"]["n_relationships"] == 1


# ---- fusion: semantic baseline + memory additions --------------------------

from continuator_memory.memory.models import Entity, Memory  # noqa: E402
from continuator_memory.memory.retrieval import (  # noqa: E402
    MAX_MEMORY_ADDITIONS,
    PROTECTED_SEMANTIC,
    EntityGroup,
    MemoryRetriever,
)
from continuator_memory.models import Chunk, RetrievedChunk  # noqa: E402

FUSION_CHUNKS = [
    Chunk(f"c{i}", "c1", i, f"text {i}", (f"m{i}",)) for i in range(1, 10)
] + [Chunk("c10", "c1", 10, "text 10", ("m4", "m9"))]


class _ChunkStore:
    def get_chunks(self, conversation_id=None):
        return list(FUSION_CHUNKS)


def _semantic(*ids):
    by_id = {c.chunk_id: c for c in FUSION_CHUNKS}
    return [RetrievedChunk(r, cid, "c1", by_id[cid].source_message_ids, 1.0 - r / 10,
                           by_id[cid].text) for r, cid in enumerate(ids, start=1)]


def _mem(mid, *sources):
    return Memory(mid, "c1", "e1", "X", "p", mid, "state", tuple(sources), 0, "T", None, None,
                  True, None)


def _fuse(semantic, records, k=5):
    group = EntityGroup(Entity("e1", "c1", "X", "x"), [[m] for m, _ in records])
    scores = {m.memory_id: s for m, s in records}
    return [c.chunk_id for c in
            MemoryRetriever(_ChunkStore(), None, None)._fuse(semantic, [group], [], scores, k, "c1")]


SEM = _semantic("c1", "c2", "c3", "c4", "c5", "c6")


def test_fusion_keeps_unlinked_semantic_rank_one():
    assert _fuse(SEM, [(_mem("a", "m5"), 9.0), (_mem("b", "m7"), 8.0)])[0] == "c1"


def test_fusion_memory_link_cannot_displace_higher_ranked_semantic_chunks():
    out = _fuse(SEM, [(_mem("a", "m6"), 9.0)])
    assert out[:PROTECTED_SEMANTIC] == ["c1", "c2", "c3"]
    assert out == ["c1", "c2", "c3", "c6", "c4"]
    assert _fuse(SEM, [(_mem("a", "m2"), 9.0)]) == ["c1", "c2", "c3", "c4", "c5"]


def test_fusion_memory_adds_source_ranked_too_low_by_semantic_retrieval():
    out = _fuse(SEM, [(_mem("a", "m8"), 3.0)])
    assert "c8" in out and out.index("c8") == PROTECTED_SEMANTIC


def test_fusion_zero_relevance_records_add_nothing():
    zero = [(_mem(f"z{i}", f"m{i}"), 0.0) for i in (6, 7, 8, 9)]
    assert _fuse(SEM, zero) == ["c1", "c2", "c3", "c4", "c5"]


def test_fusion_additions_are_bounded_and_ordered_by_record_score():
    records = [(_mem("a", "m9"), 2.0), (_mem("b", "m8"), 5.0), (_mem("c", "m7"), 4.0)]
    out = _fuse(SEM, records, k=10)
    assert out[PROTECTED_SEMANTIC:PROTECTED_SEMANTIC + MAX_MEMORY_ADDITIONS] == ["c8", "c7"]
    assert "c9" not in out and "c10" not in out


def test_fusion_deduplicates_semantic_and_memory_chunks():
    sem = _semantic("c1", "c2", "c3", "c10", "c4", "c9")
    out = _fuse(sem, [(_mem("a", "m4", "m9"), 5.0), (_mem("b", "m9"), 4.0)], k=10)
    assert len(out) == len(set(out)) == 6
    assert out == ["c1", "c2", "c3", "c10", "c4", "c9"]


def test_fusion_is_deterministic_and_prefix_consistent_across_k():
    records = [(_mem("a", "m8"), 3.0), (_mem("b", "m6"), 3.0)]
    assert _fuse(SEM, records, k=10) == _fuse(SEM, records, k=10)
    assert _fuse(SEM, records, k=10)[:5] == _fuse(SEM, records, k=5)


def test_memory_pipeline_keeps_semantic_order_when_no_record_is_relevant(pipeline):
    q = "What is the capital of Peru?"
    assert pipeline.retrieve(q, 4, "c1") == pipeline.base.retrieve(q, 4, "c1")
