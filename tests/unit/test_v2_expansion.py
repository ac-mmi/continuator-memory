import json
from collections import namedtuple

import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.expansion import (
    INCOMING,
    ORIGIN_EXPANDED,
    ORIGIN_RETRIEVED,
    OUTGOING,
    POLICY,
    SKIP_CROSS_CONVERSATION,
    SKIP_HISTORY_NOT_REQUESTED,
    SKIP_NO_KNOWLEDGE,
    SKIP_OVER_BUDGET,
    SKIP_POLICY_NEVER,
    ExpansionConfig,
    expand,
)
from continuator_memory.v2.expansion_store import build_knowledge_stored, stored_knowledge
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.relationships import ORIGIN_EXPLICIT, MemoryRelationship
from tests.helpers import HashingEmbedder

Seed = namedtuple("Seed", "rank memory_id conversation_id score")


class ScriptedClient:
    model_name = "scripted-atomizer"

    def __init__(self, claims, relations=None):
        self.claims, self.relations = claims, relations or {}

    def complete_json(self, system, prompt, schema, examples=()):
        ids = schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"]
        entries = []
        for mid in ids:
            claims = [{"claim_id": f"{mid}:c{i}", "content": content, "entities": list(entities), "qualifiers": [],
                       "temporal": {"expression": "", "resolved_date": ""}, "polarity": "positive",
                       "claim_type": ctype[0] if ctype else "", "source_message_id": mid}
                      for i, (content, entities, *ctype) in enumerate(self.claims.get(mid, ()), 1)]
            relations = [{"from_claim_id": a, "to_claim_id": b, "type": t, "evidence": ev, "source_message_id": mid}
                         for a, b, t, ev in self.relations.get(mid, ())]
            entries.append({"source_message_id": mid, "status": "claims" if claims else "no_durable_assertion",
                            "claims": claims, "explicit_relations": relations})
        return json.dumps({"messages": entries})


def conversation(cid, *messages):
    return parse_conversation({"conversation_id": cid, "messages": [
        {"id": f"m{i}", "role": role, "content": text} for i, (role, text) in enumerate(messages, 1)]})


def user(*texts):
    return [("user", t) for t in texts]


CASES = {
    "atlas": (user("Sam works at OpenAI.", "Sam joined OpenAI in 2024.", "The Atlas budget was €20,000.",
                   "The Atlas budget increased by €15,000.", "So the Atlas budget is now €35,000."),
              {"m1": [("Sam works at OpenAI.", ["Sam", "OpenAI"])],
               "m2": [("Sam joined OpenAI in 2024.", ["Sam", "OpenAI"])],
               "m3": [("The Atlas budget was €20,000.", ["Atlas"])],
               "m4": [("The Atlas budget increased by €15,000.", ["Atlas"])],
               "m5": [("The Atlas budget is now €35,000.", ["Atlas"])]},
              {"m5": [("m3:c1", "m5:c1", "SUPPORTS", "So"), ("m4:c1", "m5:c1", "SUPPORTS", "So")]}),
    "update": (user("I live in Lisbon.", "Actually, I moved to Braga."),
               {"m1": [("The user lives in Lisbon.", ["Lisbon"], "residence")],
                "m2": [("The user moved to Braga.", ["Braga"], "residence")]},
               {"m2": [("m2:c1", "m1:c1", "UPDATES", "Actually")]}),
    "supersede": (user("I live in Lisbon.", "I now live in Porto."),
                  {"m1": [("The user lives in Lisbon.", ["Lisbon"], "residence")],
                   "m2": [("The user now lives in Porto.", ["Porto"], "residence")]}, {}),
    "contradict": (user("The meeting is on Monday.", "Actually, the meeting is on Tuesday."),
                   {"m1": [("The meeting is on Monday.", ["meeting"])],
                    "m2": [("The meeting is on Tuesday.", ["meeting"])]},
                   {"m2": [("m2:c1", "m1:c1", "CONTRADICTS", "Actually")]}),
    "refers": (user("I have a cat called Miso.", "Miso is ten years old."),
               {"m1": [("The user has a cat called Miso.", ["Miso"])], "m2": [("Miso is ten years old.", ["Miso"])]},
               {"m2": [("m2:c1", "m1:c1", "REFERS_TO", "Miso")]}),
    "pets": ([("user", "Hi, my name is Sam. I have a dog called Pepper."),
              ("assistant", "Hello Sam, my name is Alex. I have a dog called Biscuit."),
              ("user", "Pepper and Biscuit should meet.")],
             {"m1": [("Sam has a dog called Pepper.", ["Sam", "Pepper"]), ("Sam's dog is named Pepper.", ["Pepper"])],
              "m2": [("Alex has a dog called Biscuit.", ["Alex", "Biscuit"])]}, {}),
}


@pytest.fixture(scope="module")
def kb():
    store = Store(":memory:")
    for cid, (messages, claims, relations) in CASES.items():
        report = V2IngestionPipeline(store, SemanticAtomizer(ScriptedClient(claims, relations)), targets_per_window=2,
                                     context_before=2).ingest(conversation(cid, *messages))
        assert not report.issues, report.issues
        build_knowledge_stored(store.connection, cid)
    return store, {cid: stored_knowledge(store.connection, cid) for cid in CASES}


def seeds(cid, *ids):
    return [Seed(rank, f"mem:{m}", cid, 1.0 - rank / 10) for rank, m in enumerate(ids, 1)]


def added(ctx):
    return [i.memory_id for i in ctx.expanded]


def roles(ctx):
    return [(i.memory_id, i.paths[0].type, i.paths[0].direction, i.paths[0].role) for i in ctx.expanded]


# ---- configuration and policy ---------------------------------------------------------------


def test_config_allows_one_hop_only_and_validates_budgets():
    assert ExpansionConfig().to_dict() == {"max_depth": 1, "max_expanded_memories": 5, "max_context_items": None,
                                           "types": ["UPDATES", "SUPERSEDES", "CONTRADICTS", "EXTENDS", "DERIVES",
                                                     "REFERS_TO"]}
    for bad in ({"max_depth": 2}, {"max_depth": 0}, {"max_expanded_memories": -1}, {"max_context_items": 0},
                {"types": ("PROFILE",)}):
        with pytest.raises(ValueError):
            ExpansionConfig(**bad)


def test_policy_covers_every_type_in_both_directions():
    types = {"UPDATES", "SUPERSEDES", "CONTRADICTS", "EXTENDS", "DERIVES", "REFERS_TO"}
    assert set(POLICY) == {(t, d) for t in types for d in (OUTGOING, INCOMING)}


# ---- per-type policy and direction ----------------------------------------------------------


def test_extends_adds_the_other_side_in_both_directions(kb):
    _, k = kb
    assert roles(expand("When did Sam join OpenAI?", seeds("atlas", "m2:c1"), k)) == [
        ("mem:m1:c1", "EXTENDS", OUTGOING, "extended")]
    assert roles(expand("Where does Sam work?", seeds("atlas", "m1:c1"), k)) == [
        ("mem:m2:c1", "EXTENDS", INCOMING, "extension")]


def test_derives_adds_supporting_memories_and_the_conclusion(kb):
    _, k = kb
    ctx = expand("What is the Atlas budget now?", seeds("atlas", "m5:c1"), k)
    assert roles(ctx) == [("mem:m3:c1", "DERIVES", OUTGOING, "support"), ("mem:m4:c1", "DERIVES", OUTGOING, "support")]
    assert roles(expand("What was the Atlas budget?", seeds("atlas", "m3:c1"), k)) == [
        ("mem:m5:c1", "DERIVES", INCOMING, "conclusion")]


def test_updates_adds_the_replacement_and_history_only_when_asked(kb):
    _, k = kb
    assert roles(expand("Where did the user live?", seeds("update", "m1:c1"), k)) == [
        ("mem:m2:c1", "UPDATES", INCOMING, "current_state")]
    now = expand("Where does the user live?", seeds("update", "m2:c1"), k)
    assert added(now) == [] and now.diagnostics["skipped"] == {SKIP_HISTORY_NOT_REQUESTED: 1}
    assert roles(expand("Where did the user live before?", seeds("update", "m2:c1"), k)) == [
        ("mem:m1:c1", "UPDATES", OUTGOING, "history")]


def test_supersedes_adds_the_replacement_and_not_all_history(kb):
    _, k = kb
    assert roles(expand("Where does the user live?", seeds("supersede", "m1:c1"), k)) == [
        ("mem:m2:c1", "SUPERSEDES", INCOMING, "current_state")]
    assert added(expand("Where does the user live?", seeds("supersede", "m2:c1"), k)) == []


def test_contradicts_is_included_from_either_side_and_marked(kb):
    _, k = kb
    for seed, other, direction in (("m2:c1", "mem:m1:c1", OUTGOING), ("m1:c1", "mem:m2:c1", INCOMING)):
        ctx = expand("When is the meeting?", seeds("contradict", seed), k)
        assert roles(ctx) == [(other, "CONTRADICTS", direction, "conflict")]
        assert "conflict via mem:m2:c1 --CONTRADICTS--> mem:m1:c1" in ctx.expanded[0].format()


def test_refers_to_follows_the_outgoing_reference_only(kb):
    _, k = kb
    assert roles(expand("How old is Miso?", seeds("refers", "m2:c1"), k)) == [
        ("mem:m1:c1", "REFERS_TO", OUTGOING, "reference")]
    back = expand("What cat does the user have?", seeds("refers", "m1:c1"), k)
    assert added(back) == [] and back.diagnostics["skipped"] == {SKIP_POLICY_NEVER: 1}


def test_type_filter_disables_expansion_per_type(kb):
    _, k = kb
    ctx = expand("What is the Atlas budget now?", seeds("atlas", "m5:c1"), k, ExpansionConfig(types=("EXTENDS",)))
    assert added(ctx) == [] and ctx.diagnostics["skipped"] == {"type_not_expandable": 2}


# ---- ranking, budget, deduplication ---------------------------------------------------------


def test_seeds_are_kept_in_order_and_expansions_follow_the_documented_key(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m5:c1", "m2:c1"), k)
    assert [(i.memory_id, i.origin, i.rank, i.depth) for i in ctx.memories] == [
        ("mem:m5:c1", ORIGIN_RETRIEVED, 1, 0), ("mem:m2:c1", ORIGIN_RETRIEVED, 2, 0),
        ("mem:m3:c1", ORIGIN_EXPANDED, 3, 1), ("mem:m4:c1", ORIGIN_EXPANDED, 4, 1), ("mem:m1:c1", ORIGIN_EXPANDED, 5, 1)]
    ranked = [Seed(3, "mem:m4:c1", "atlas", 0.2), Seed(4, "mem:m3:c1", "atlas", 0.1)]
    ctx = expand("q", seeds("atlas", "m5:c1"), k, retrieved=ranked)
    assert added(ctx) == ["mem:m4:c1", "mem:m3:c1"]
    assert [(i.retrieval_rank, i.retrieval_score) for i in ctx.expanded] == [(3, 0.2), (4, 0.1)]


def test_max_expansion_and_context_truncation_are_deterministic_and_reported(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m5:c1", "m2:c1"), k, ExpansionConfig(max_expanded_memories=1))
    assert added(ctx) == ["mem:m3:c1"] and ctx.diagnostics["skipped"] == {SKIP_OVER_BUDGET: 2}
    ctx = expand("q", seeds("atlas", "m5:c1", "m2:c1", "m1:c1"), k, ExpansionConfig(max_context_items=2))
    assert [i.memory_id for i in ctx.memories] == ["mem:m5:c1", "mem:m2:c1"]
    assert ctx.diagnostics["truncated"] and ctx.diagnostics["truncated_memories"] == [
        "atlas/mem:m1:c1", "atlas/mem:m3:c1", "atlas/mem:m4:c1"]
    assert expand("q", seeds("atlas", "m5:c1"), k, ExpansionConfig(max_expanded_memories=0)).expanded == ()


def test_memory_reached_by_two_paths_appears_once_with_both_paths(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m3:c1", "m4:c1"), k)
    assert added(ctx) == ["mem:m5:c1"]
    assert [p.seed_memory_id for p in ctx.expanded[0].paths] == ["mem:m3:c1", "mem:m4:c1"]
    assert ctx.diagnostics["duplicate_paths"] == 1 and ctx.diagnostics["candidate_paths"] == 2


def test_a_neighbour_that_is_already_a_seed_is_not_added_again(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m5:c1", "m3:c1"), k)
    assert [i.memory_id for i in ctx.memories] == ["mem:m5:c1", "mem:m3:c1", "mem:m4:c1"]
    assert [p.edge for p in ctx.memories[1].paths] == ["mem:m5:c1 --DERIVES--> mem:m3:c1"]
    assert ctx.diagnostics["paths_to_seeds"] == 2  # m5 -> m3 and m3 <- m5 conclusion


# ---- source windows, speakers, entities, provenance -----------------------------------------


def test_source_windows_are_deduplicated_and_keep_every_citing_memory(kb):
    _, k = kb
    ctx = expand("q", seeds("pets", "m1:c1", "m1:c2"), k)
    assert len(ctx.source_windows) == 1 and ctx.diagnostics["source_windows"] == 1
    assert ctx.source_windows[0].memory_ids == ("pets/mem:m1:c1", "pets/mem:m1:c2")
    ctx = expand("q", seeds("atlas", "m2:c1"), k)  # m2 window [m1, m2]; m1 window [m1]
    assert [s.window.message_ids for s in ctx.source_windows] == [("m1", "m2"), ("m1",)]
    assert ctx.diagnostics["source_messages"] == 2


def test_speakers_are_preserved_and_never_collapsed(kb):
    _, k = kb
    ctx = expand("Whose dog is Biscuit?", seeds("pets", "m1:c1", "m2:c1"), k)
    speakers = {m.message_id: m.speaker for s in ctx.source_windows for m in s.messages}
    assert speakers == {"m1": "Sam (user)", "m2": "Alex (assistant)"}
    text = ctx.format()
    assert "[m1] Sam (user): Hi, my name is Sam." in text and "[m2] Alex (assistant): Hello Sam" in text
    assert "asserted by Alex (assistant)" in text and "the user" not in text.lower()


def test_unnamed_speakers_get_role_labels(kb):
    _, k = kb
    ctx = expand("q", seeds("update", "m2:c1"), k)
    assert {m.speaker for s in ctx.source_windows for m in s.messages} == {"User"}


def test_entities_and_provenance_are_preserved(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m2:c1"), k)
    item = ctx.expanded[0].to_dict()
    assert item["source_claim_ids"] == ["m1:c1"] and item["source_message_ids"] == ["m1"]
    assert item["paths"][0]["relationship_id"] == "rel:atlas/mem:m2:c1/EXTENDS/mem:m1:c1"
    assert {(e["canonical_name"], tuple(e["memories"])) for e in ctx.entities} >= {
        ("Sam", ("atlas/mem:m2:c1", "atlas/mem:m1:c1")), ("OpenAI", ("atlas/mem:m2:c1", "atlas/mem:m1:c1"))}
    assert all(r["entity_id"] for i in ctx.to_dict()["memories"] for r in i["entity_refs"])


# ---- isolation, cycles, depth, determinism --------------------------------------------------


def test_expansion_never_crosses_conversations(kb):
    _, k = kb
    ctx = expand("q", seeds("atlas", "m2:c1") + [Seed(2, "mem:m2:c1", "update", 0.5)], k)
    assert {i.conversation_id for i in ctx.memories if i.origin == ORIGIN_EXPANDED} == {"atlas"}
    assert all(p.seed_memory_id and i.conversation_id == ctx.memories[0].conversation_id
               for i in ctx.expanded for p in i.paths)
    assert expand("q", seeds("unknown", "m1:c1"), k).diagnostics["skipped"] == {SKIP_NO_KNOWLEDGE: 1}


def test_a_foreign_edge_is_rejected_and_reported(kb):
    store, _ = kb
    knowledge = stored_knowledge(store.connection, "atlas")
    foreign = MemoryRelationship("update", "mem:m2:c1", "mem:m1:c1", "EXTENDS", "m2:c1", origin=ORIGIN_EXPLICIT,
                                 reason="injected")
    knowledge.graph._additive[foreign.key] = foreign
    ctx = expand("q", seeds("atlas", "m2:c1"), {"atlas": knowledge})
    assert ctx.diagnostics["skipped"][SKIP_CROSS_CONVERSATION] == 1
    assert ctx.diagnostics["cross_conversation_violations"] and added(ctx) == ["mem:m1:c1"]


def chain(store):
    knowledge = stored_knowledge(store.connection, "atlas")
    for src, dst, t in (("m1:c1", "m3:c1", "EXTENDS"), ("m3:c1", "m4:c1", "EXTENDS"), ("m4:c1", "m1:c1", "DERIVES")):
        assert knowledge.graph.add(MemoryRelationship("atlas", f"mem:{src}", f"mem:{dst}", t, src,
                                                      origin=ORIGIN_EXPLICIT, reason="test chain")) is None
    return {"atlas": knowledge}


def test_one_hop_boundary_and_cycle_safety(kb):
    store, _ = kb
    k = chain(store)  # m1 -EXTENDS-> m3 -EXTENDS-> m4 -DERIVES-> m1 (a cycle across types)
    ctx = expand("q", seeds("atlas", "m3:c1"), k)
    assert sorted(added(ctx)) == ["mem:m1:c1", "mem:m4:c1", "mem:m5:c1"]  # direct neighbours only; m2 is 2 hops
    ctx = expand("q", seeds("atlas", "m1:c1"), k)
    assert "mem:m3:c1" in added(ctx) and all(p.depth == 1 for i in ctx.expanded for p in i.paths)
    assert ctx.diagnostics["relationships_traversed"] == len(ctx.relationships) == 3


def test_serialization_is_byte_identical_across_independent_runs():
    def run():
        store = Store(":memory:")
        knowledge = {}
        for cid, (messages, claims, relations) in CASES.items():
            V2IngestionPipeline(store, SemanticAtomizer(ScriptedClient(claims, relations)), targets_per_window=2,
                                context_before=2).ingest(conversation(cid, *messages))
            knowledge[cid] = build_knowledge_stored(store.connection, cid)
        out = [expand(q, seeds(cid, *ids), knowledge).to_dict() for q, cid, ids in (
            ("q", "atlas", ("m5:c1", "m2:c1")), ("Where did the user live?", "update", ("m2:c1",)),
            ("q", "pets", ("m2:c1", "m1:c2")))]
        return json.dumps(out, sort_keys=True, ensure_ascii=False)

    assert run() == run()


def test_expansion_consumes_real_retrieval_results_unchanged(kb):
    store, k = kb
    retriever = MemoryRetriever(HashingEmbedder(), RetrievalConfig(min_similarity=None))
    retriever.index_store(MemoryStore(store.connection), "atlas")
    outcome = retriever.search("What is the Atlas budget now?", 1, "atlas")
    before = [r.to_dict() for r in outcome.results]
    ctx = expand(outcome.question, outcome.results, k)
    assert [r.to_dict() for r in outcome.results] == before
    assert [i.memory_id for i in ctx.memories] == ["mem:m5:c1", "mem:m3:c1", "mem:m4:c1"]
    assert ctx.seeds[0].retrieval_score == outcome.results[0].score
