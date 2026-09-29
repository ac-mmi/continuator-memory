import pytest

from continuator_memory.memory.context import render_memory_context
from continuator_memory.memory.extraction import MemoryExtractor
from continuator_memory.memory.pipeline import MemoryPipeline
from continuator_memory.memory.retrieval import (
    DIRECT_QUESTION_MATCH,
    ENTITY_PLUS_ATTRIBUTE_MATCH,
    RELATIONSHIP_SUPPORT,
    STATE_HISTORY,
    MemoryRetriever,
)
from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from tests.helpers import ScriptedJSONClient, doc, mem, rel

TEAM = [
    "Omar is from Cairo.",
    "Omar's birthday is on September 8.",
    "Ines calls her dog Rex.",
    "The Project Nova launch date is June 3.",
    "The Project Nova launch moved from June 3 to July 1.",
    "Sam has a dog called Pepper.",
    "Priya handles the travel budget.",
    "Priya is from Oslo.",
]
PETS = ["Jonas has two cats named Pixel and Byte."]
SCRIPT = {
    TEAM[0]: doc([mem("Omar", "origin", "Cairo")]),
    TEAM[1]: doc([mem("Omar", "birthday", "September 8")]),
    TEAM[2]: doc([mem("Ines", "pet_name", "Rex")]),
    TEAM[3]: doc([mem("Project Nova", "launch_date", "June 3")]),
    TEAM[4]: doc([mem("Project Nova", "launch_date", "July 1", "update", "June 3")]),
    TEAM[5]: doc([mem("Sam", "has_dog", "Pepper")]),
    TEAM[6]: doc(relationships=[rel("Priya", "handles", "travel budget")]),
    TEAM[7]: doc([mem("Priya", "origin", "Oslo")]),
    PETS[0]: doc([mem("Jonas", "has_cats", "Pixel and Byte")]),
}


def _conversation(cid, texts):
    return parse_conversation({"conversation_id": cid, "messages": [
        {"id": f"m{i + 1}", "role": "user", "content": t} for i, t in enumerate(texts)]})


def _build(store, embedder):
    p = MemoryPipeline(store, embedder, None, MemoryExtractor(ScriptedJSONClient(SCRIPT)),
                       window_size=1, overlap=0)
    p.ingest(_conversation("team", TEAM))
    p.ingest(_conversation("pets", PETS))
    return p


@pytest.fixture
def pipeline(store, embedder):
    return _build(store, embedder)


def _context(r):
    return {(m.subject, m.predicate, m.value) for g in r.groups for s in g.slots for m in s}


def _candidates(r):
    return {(m.subject, m.predicate, m.value)
            for g in r.candidate_groups for s in g.slots for m in s}


def _reasons(r):
    return {(e["entity"], e["predicate"], e["value"]): e["reason"] for e in r.selection}


def test_entity_only_match_does_not_enter_memory_context(pipeline):
    r = pipeline.retrieve_memory("What is Omar's last name?", 3, "team")
    assert [e.name for e in r.analysis.entities] == ["Omar"]
    assert ("Omar", "origin", "Cairo") in _candidates(r)
    assert not any(subject == "Omar" for subject, _, _ in _context(r))


def test_generic_word_only_match_does_not_enter(pipeline):
    r = pipeline.retrieve_memory("What is Omar's last name?", 3, "team")
    assert ("Ines", "pet_name", "Rex") in _candidates(r)
    assert ("Ines", "pet_name", "Rex") not in _context(r)
    assert _context(r) == set() and r.selection == []
    assert "(no stored facts match this question)" in render_memory_context(r)


def test_predicate_value_match_enters(pipeline):
    r = pipeline.retrieve_memory("What is the launch date?", 3, "team")
    assert r.analysis.entities == ()
    assert _reasons(r)[("Project Nova", "launch_date", "July 1")] == DIRECT_QUESTION_MATCH
    entry = next(e for e in r.selection if e["value"] == "July 1")
    assert entry["evidence"] == ["date", "launch"] and entry["score"] > 0
    assert set(entry) == {"record_id", "entity", "predicate", "value", "is_current", "score",
                          "reason", "evidence"}


def test_entity_plus_relevant_predicate_enters(pipeline):
    r = pipeline.retrieve_memory("When is Omar's birthday?", 3, "team")
    assert _context(r) == {("Omar", "birthday", "September 8")}
    assert _reasons(r)[("Omar", "birthday", "September 8")] == ENTITY_PLUS_ATTRIBUTE_MATCH


def test_irrelevant_memory_from_another_conversation_does_not_enter(pipeline):
    r = pipeline.retrieve_memory("What are the names of Sam's cats?", 3, None)
    assert ("Jonas", "has_cats", "Pixel and Byte") in _candidates(r)
    assert _context(r) == set()
    assert "Pixel" not in render_memory_context(r)


def test_current_relevant_memory_remains_included(pipeline):
    r = pipeline.retrieve_memory("What is the current launch date of Project Nova?", 3, "team")
    (group,) = r.groups
    assert group.slots[0][0].value == "July 1" and group.slots[0][0].is_current
    assert render_memory_context(r).startswith("MEMORY:\nProject Nova\n- launch_date: July 1 [CURRENT")


def test_superseded_state_behavior_unchanged(pipeline):
    for q in ("When does the Project Nova launch date fall now?",
              "What was the original launch date of Project Nova?"):
        r = pipeline.retrieve_memory(q, 3, "team")
        (context_slot,) = [s for g in r.groups for s in g.slots]
        (candidate_slot,) = [s for g in r.candidate_groups for s in g.slots
                             if s[0].predicate == "launch_date"]
        assert context_slot == candidate_slot
        assert [(m.value, m.is_current) for m in context_slot] == [("July 1", True),
                                                                   ("June 3", False)]
        assert "June 3 [SUPERSEDED by July 1" in render_memory_context(r)
    r = pipeline.retrieve_memory("Why did the launch move to July 1?", 3, "team")
    reasons = _reasons(r)
    assert reasons[("Project Nova", "launch_date", "July 1")] == DIRECT_QUESTION_MATCH
    assert reasons[("Project Nova", "launch_date", "June 3")] == STATE_HISTORY


def test_relationship_support_keeps_one_hop_memories(pipeline):
    r = pipeline.retrieve_memory("Which city is the person who handles the travel budget from?",
                                 3, "team")
    assert [(x.subject, x.predicate, x.object) for x in r.relationships] == [
        ("Priya", "handles", "travel budget")]
    assert _reasons(r)[("Priya", "origin", "Oslo")] == RELATIONSHIP_SUPPORT


QUESTIONS = [
    ("What is Omar's last name?", "team"),
    ("When is Omar's birthday?", "team"),
    ("What is the launch date?", "team"),
    ("What was the original launch date of Project Nova?", "team"),
    ("What are the names of Sam's cats?", None),
    ("Which city is the person who handles the travel budget from?", "team"),
]


def test_memory_context_is_deterministic(pipeline, embedder):
    with Store(":memory:") as fresh:
        other = _build(fresh, embedder)
        for q, scope in QUESTIONS:
            a = pipeline.retrieve_memory(q, 3, scope)
            b = pipeline.retrieve_memory(q, 3, scope)
            c = other.retrieve_memory(q, 3, scope)
            assert a.trace() == b.trace() == c.trace()
            assert render_memory_context(a) == render_memory_context(c)


def test_semantic_excerpt_retrieval_completely_unchanged(pipeline, monkeypatch):
    cases = [(q, scope, k) for q, scope in QUESTIONS for k in (2, 5)]
    filtered = [pipeline.retrieve_memory(q, k, scope) for q, scope, k in cases]

    def unfiltered(self, analysis, groups, relationships, scores, ent_by_id):
        return groups, relationships, [], {}

    monkeypatch.setattr(MemoryRetriever, "_select_context", unfiltered)
    for (q, scope, k), after in zip(cases, filtered):
        before = pipeline.retrieve_memory(q, k, scope)
        assert after.chunks == before.chunks
        assert after.candidate_groups == before.groups
        assert after.candidate_relationships == before.relationships
    assert any(len(r.groups) < len(r.candidate_groups) for r in filtered)


def test_phase33_formation_statistics_unchanged(pipeline, embedder):
    stats = pipeline.memory.stats()
    reports = [r.to_dict() for r in pipeline.reports]
    for q, scope in QUESTIONS:
        pipeline.retrieve_memory(q, 5, scope)
    assert pipeline.memory.stats() == stats
    assert stats["memories"] == 8 and stats["memories_current"] == 7
    assert stats["relationships"] == 1
    assert stats["consolidation_actions"]["memory"] == {"CREATE": 7, "SUPERSEDE": 1}

    def strip(d):
        return {k: v for k, v in d.items() if not k.endswith("_ms")}

    with Store(":memory:") as fresh:
        other = _build(fresh, embedder)
        assert strip(other.memory.stats()) == strip(stats)
        assert [strip(r.to_dict()) for r in other.reports] == [strip(r) for r in reports]
