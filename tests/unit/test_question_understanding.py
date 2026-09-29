import json

import pytest

import continuator_memory.memory.query as query
from continuator_memory.memory.context import build_memory_prompt, render_memory_context
from continuator_memory.memory.extraction import MemoryExtractor
from continuator_memory.memory.models import Entity
from continuator_memory.memory.pipeline import MemoryPipeline
from continuator_memory.memory.query import HISTORICAL_MARKERS, LexicalQuestionAnalyzer
from continuator_memory.memory.retrieval import (
    CONSTRAINT_MATCH,
    CONSTRAINT_PIVOT,
    ENTITY_NAMED,
)
from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from tests.helpers import PHASE2_BENCHMARK, ScriptedJSONClient, doc, mem

OMAR = Entity("c:ent1", "c", "Omar", "omar")
SAM = Entity("c:ent2", "c", "Sam", "sam")

MH05 = "Which city is the person who handles the offsite budget from?"
MH06 = "Which city is the team member whose birthday falls during the offsite from?"

TEAM = [
    "Omar is the backend developer. He's from Cairo and he's vegetarian.",
    "Mei is the product manager, originally from Taipei. She plays the violin in a community orchestra.",
    "Mei will handle the budget, and Jonas volunteered to book the train tickets.",
    "Omar's birthday falls during the offsite, on September 8th, so let's plan a small cake.",
    "Jonas does QA. He's from Hamburg and has two cats named Pixel and Byte.",
    "The Budget Committee meets in Room 4.",
]
SCRIPT = {
    TEAM[0]: doc([mem("Omar", "role", "backend developer"), mem("Omar", "origin", "Cairo"),
                  mem("Omar", "dietary_preferences", "vegetarian")]),
    TEAM[1]: doc([mem("Mei", "role", "product manager"), mem("Mei", "origin", "Taipei"),
                  mem("Mei", "plays_the_violin", "community orchestra")]),
    TEAM[2]: doc([mem("Mei", "handles", "budget"), mem("Jonas", "booked_tickets", "train tickets")]),
    TEAM[3]: doc([mem("Omar", "birthday", "September 8th")]),
    TEAM[4]: doc([mem("Jonas", "origin", "Hamburg"), mem("Jonas", "has_cats", "Pixel and Byte")]),
    TEAM[5]: doc([mem("Budget Committee", "location", "Room 4")]),
}


def _analyze(question, entities=()):
    return LexicalQuestionAnalyzer().analyze(question, list(entities), set(), set())


def _pipeline(store, embedder, texts, script, window_size=2, overlap=0):
    p = MemoryPipeline(store, embedder, None, MemoryExtractor(ScriptedJSONClient(script)),
                       window_size=window_size, overlap=overlap)
    p.ingest(parse_conversation({"conversation_id": "team", "messages": [
        {"id": f"m{i + 1:02d}", "role": "user", "content": t} for i, t in enumerate(texts)]}))
    return p


@pytest.fixture
def team(store, embedder):
    return _pipeline(store, embedder, TEAM, SCRIPT)


def _context(r):
    return [(g.entity.name, m.predicate, m.value) for g in r.groups for s in g.slots for m in s]


# ---- analysis ---------------------------------------------------------------

def test_explicit_target_and_requested_attribute():
    a = _analyze("What is Omar's role?", [OMAR, SAM])
    assert a.targets == (OMAR,) and a.constraints == ()
    assert a.requested_attribute.tokens == frozenset({"role"})
    b = _analyze("What is Sam's salary?", [OMAR, SAM])
    assert b.targets == (SAM,) and b.requested_attribute.tokens == frozenset({"salary"})
    assert _analyze("What is Omar's last name?", [OMAR]).requested_attribute.text == "last name"


def test_requested_attribute_is_distinct_from_constraint():
    a = _analyze(MH05)
    (c,) = a.constraints
    assert a.requested_attribute.tokens == frozenset({"city"})
    assert a.requested_attribute.frame == "from"
    assert c.marker == "who" and c.head == "handle"
    assert c.terms == frozenset({"handle", "offsite", "budget"})
    assert not a.requested_attribute.tokens & c.terms
    (d,) = _analyze(MH06).constraints
    assert d.marker == "whose" and d.head == "birthday" and "city" not in d.terms


def test_temporal_scope_unchanged():
    for q in [MH05, MH06, "What is Omar's role?", "What was the original launch date?",
              "Who was the team lead before Hana took over?", "What is the current budget?"]:
        words = set(query.normalize_value(q).split())
        expected = "historical" if words & HISTORICAL_MARKERS else "current"
        assert _analyze(q).temporal_scope == expected


def test_analysis_is_deterministic():
    for q in [MH05, MH06, "What is Omar's role?", "Where is Omar from?"]:
        a, b = _analyze(q, [OMAR, SAM]), _analyze(q, [OMAR, SAM])
        assert a == b and a.to_dict() == b.to_dict()
        assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


# ---- selection --------------------------------------------------------------

def test_handles_budget_constraint_identifies_mei(team):
    r = team.retrieve_memory(MH05, 3, "team")
    u = r.trace()["question_understanding"]
    assert [(m["entity"], m["predicate"], m["value"]) for m in u["constraint_matches"]] == [
        ("Mei", "handles", "budget")]
    assert [p["entity"] for p in u["pivots"]] == ["Mei"]
    assert u["entity_roles"]["Mei"] == CONSTRAINT_PIVOT
    assert ("Mei", "origin", "Taipei") in _context(r)
    origin = next(e for e in r.selection if e["value"] == "Taipei")
    assert origin["reason"] == CONSTRAINT_PIVOT and origin["evidence"] == ["frame:from Taipei"]


def test_birthday_constraint_identifies_unnamed_omar(team):
    r = team.retrieve_memory(MH06, 3, "team")
    assert "Omar" not in [e.name for e in r.analysis.targets]
    u = r.trace()["question_understanding"]
    assert [p["entity"] for p in u["pivots"]] == ["Omar"]
    reasons = {(e["predicate"], e["value"]): e["reason"] for e in r.selection}
    assert reasons[("birthday", "September 8th")] == CONSTRAINT_MATCH
    assert reasons[("origin", "Cairo")] == CONSTRAINT_PIVOT


def test_constraint_pivot_does_not_expose_unrelated_pivot_slots(team):
    r = team.retrieve_memory(MH05, 3, "team")
    mei = [(p, v) for e, p, v in _context(r) if e == "Mei"]
    assert sorted(mei) == [("handles", "budget"), ("origin", "Taipei")]
    candidates = {m.predicate for g in r.candidate_groups if g.entity.name == "Mei"
                  for s in g.slots for m in s}
    assert {"role", "plays_the_violin"} <= candidates


def test_entity_only_matching_does_not_become_constraint_matching(team):
    q = "Which room is the group that budgets the offsite in?"
    r = team.retrieve_memory(q, 3, "team")
    assert r.analysis.constraints[0].head == "budget"
    u = r.trace()["question_understanding"]
    assert u["constraint_matches"] == [] and u["pivots"] == []
    assert not any(e["reason"] in (CONSTRAINT_MATCH, CONSTRAINT_PIVOT) for e in r.selection)


def test_direct_question_without_constraint_uses_phase34_selection(team, monkeypatch):
    questions = ["When is Omar's birthday?", "What is Omar's role?", "Where is Omar from?",
                 "Who booked the train tickets?"]
    new = [team.retrieve_memory(q, 3, "team") for q in questions]
    assert all(r.analysis.constraints == () for r in new)
    monkeypatch.setattr(query, "parse_structure", lambda question, targets: ((), None))
    for q, r in zip(questions, new):
        old = team.retrieve_memory(q, 3, "team")
        assert r.groups == old.groups and r.relationships == old.relationships
        assert r.selection == old.selection and r.chunks == old.chunks
        assert render_memory_context(r) == render_memory_context(old)
        assert r.trace()["question_understanding"]["pivots"] == []
    roles = team.retrieve_memory("What is Omar's role?", 3, "team")
    assert roles.trace()["question_understanding"]["entity_roles"] == {"Omar": ENTITY_NAMED}


def test_constraint_pivot_leaves_excerpts_unchanged(team, monkeypatch):
    new = [team.retrieve_memory(q, k, "team") for q in (MH05, MH06) for k in (2, 5)]
    monkeypatch.setattr(query, "parse_structure", lambda question, targets: ((), None))
    old = [team.retrieve_memory(q, k, "team") for q in (MH05, MH06) for k in (2, 5)]
    for a, b in zip(new, old):
        assert a.chunks == b.chunks and a.candidate_groups == b.candidate_groups
        assert len(_context(a)) > len(_context(b))


# ---- integration: benchmark team-offsite conversation -----------------------

def _team_offsite():
    data = json.loads(PHASE2_BENCHMARK.read_text())
    conv = next(c for c in data["conversations"] if c["conversation_id"] == "team-offsite")
    by_id = {m["id"]: m["content"] for m in conv["messages"]}
    script = {
        by_id["m03"]: doc([mem("Ines", "role", "designer"), mem("Ines", "origin", "Lisbon")]),
        by_id["m05"]: doc([mem("Omar", "role", "backend developer"), mem("Omar", "origin", "Cairo"),
                           mem("Omar", "dietary_preferences", "vegetarian")]),
        by_id["m07"]: doc([mem("Mei", "role", "product manager"), mem("Mei", "origin", "Taipei"),
                           mem("Mei", "plays_the_violin", "community orchestra")]),
        by_id["m09"]: doc([mem("Jonas", "origin", "Hamburg"),
                           mem("Jonas", "has_cats", "Pixel and Byte")]),
        by_id["m13"]: doc([mem("Mei", "handles", "budget"),
                           mem("Jonas", "booked_tickets", "train tickets")]),
        by_id["m15"]: doc([mem("Omar", "birthday", "September 8th")]),
    }
    return parse_conversation(conv), script


@pytest.mark.parametrize("question, pivot, answer_line", [
    (MH05, "Mei", "- origin: Taipei [CURRENT]"),
    (MH06, "Omar", "- origin: Cairo [CURRENT]"),
])
def test_integration_team_offsite_multihop(embedder, question, pivot, answer_line):
    conv, script = _team_offsite()
    with Store(":memory:") as store:
        p = MemoryPipeline(store, embedder, None, MemoryExtractor(ScriptedJSONClient(script)),
                           window_size=4, overlap=2)
        p.ingest(conv)
        r = p.retrieve_memory(question, 5, "team-offsite")
        u = r.trace()["question_understanding"]
        assert [x["entity"] for x in u["pivots"]] == [pivot]
        memory_block = render_memory_context(r)
        assert f"\n{pivot}\n" in memory_block + "\n" and answer_line in memory_block
        others = {"Ines", "Jonas", "Mei", "Omar"} - {pivot}
        assert not any(f"\n{name}\n" in memory_block + "\n" for name in others)
        prompt = build_memory_prompt(question, r)
        assert prompt.endswith(f"Question: {question}\nAnswer:")
