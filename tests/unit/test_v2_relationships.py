import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.memory import (
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    LINK_TYPES,
    LINK_UPDATES,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_SUPERSEDED,
    MemoryLink,
)
from continuator_memory.v2.memory_store import MemoryStore, consolidate_stored
from continuator_memory.v2.models import Claim, Provenance, Relation, Temporal
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.relationship_store import RelationshipStore, build_relationships_stored
from continuator_memory.v2.relationships import (
    ADDITIVE_TYPES,
    ALREADY_RELATED,
    CROSS_CONVERSATION,
    CYCLE,
    DUPLICATE,
    INVALID_ORIGIN,
    INVALID_TYPE,
    LINK_DERIVES,
    LINK_EXTENDS,
    MISSING_SOURCE,
    MISSING_TARGET,
    ORIGIN_CONSOLIDATION,
    ORIGIN_EXTENDS_RULE,
    ORIGIN_SUPPORTS,
    RELATIONSHIP_TYPES,
    SELF_LINK,
    SEMANTICS,
    MemoryRelationship,
    build_relationships,
    relationship_id,
    versioning_cycles,
)
from continuator_memory.v2.store import SemanticStore

CONV = "conv"


def claim(cid, content, position, entities=(), claim_type=None, polarity="positive", temporal=None, conv=CONV):
    mid = cid.rsplit(":c", 1)[0]
    return Claim(cid, content, Provenance(conv, mid, position, "user"), tuple(entities), (),
                 Temporal(temporal) if temporal else None, polarity, claim_type)


def rel(rid, src, dst, rtype, evidence="evidence"):
    return Relation(rid, src, dst, rtype, rid.rsplit(":r", 1)[0], evidence)


def build(claims, relations=(), conv=CONV):
    c = consolidate(conv, claims, relations)
    before = [m.to_dict() for m in c.memories.values()], list(c.links), [d.to_dict() for d in c.decisions]
    graph = build_relationships(conv, c.memories.values(), c.links, c.sources, c.decisions, claims, relations)
    after = [m.to_dict() for m in c.memories.values()], list(c.links), [d.to_dict() for d in c.decisions]
    assert before == after  # the relationship layer never changes consolidation state
    return c, graph


def edges(graph, type_=None):
    return [(r.source_memory_id, r.type, r.target_memory_id) for r in graph.edges if type_ in (None, r.type)]


WORKS = claim("m1:c1", "Sam works at OpenAI.", 0, ["Sam", "OpenAI"])
JOINED = claim("m2:c1", "Sam joined OpenAI in 2024.", 1, ["Sam", "OpenAI"], temporal="in 2024")
BUDGET = [claim("m1:c1", "The Atlas budget was €20,000.", 0, ["Atlas"]),
          claim("m2:c1", "The Atlas budget increased by €15,000.", 1, ["Atlas"]),
          claim("m3:c1", "The Atlas budget is now €35,000.", 2, ["Atlas"])]
BUDGET_SUPPORTS = [rel("m3:r1", "m1:c1", "m3:c1", "SUPPORTS", "So"), rel("m3:r2", "m2:c1", "m3:c1", "SUPPORTS", "So")]


# ---- semantics --------------------------------------------------------------


def test_semantics_table_and_existing_types_are_preserved():
    assert LINK_TYPES == ("UPDATES", "SUPERSEDES", "CONTRADICTS", "REFERS_TO")
    assert RELATIONSHIP_TYPES == (*LINK_TYPES, "EXTENDS", "DERIVES") and ADDITIVE_TYPES == ("EXTENDS", "DERIVES")
    table = {t: (s.meaning, s.replaces_target, s.target_remains_current) for t, s in SEMANTICS.items()}
    assert table == {
        "UPDATES": ("newer statement changes target", "conceptually yes", "depends on existing temporal logic"),
        "SUPERSEDES": ("newer state replaces target", "yes", "usually no"),
        "CONTRADICTS": ("conflicting statement", "no", "yes/contested"),
        "REFERS_TO": ("references another memory", "no", "yes"),
        "EXTENDS": ("adds information", "no", "yes"),
        "DERIVES": ("conclusion supported by target", "no", "yes"),
    }


# ---- EXTENDS ----------------------------------------------------------------


def test_extends_fixture_links_newer_to_older_without_changing_either():
    c, graph = build([WORKS, JOINED])
    assert edges(graph) == [("mem:m2:c1", "EXTENDS", "mem:m1:c1")]
    (edge,) = graph.additive
    assert edge.origin == ORIGIN_EXTENDS_RULE and edge.claim_id == "m2:c1" and edge.relation_id is None
    assert "sam" in edge.reason and "openai" in edge.reason
    for m in c.memories.values():
        assert m.status == STATUS_CURRENT and m.superseded_by is None and m.valid_until is None
    assert [d.action for d in c.decisions] == ["CREATE", "CREATE"]


@pytest.mark.parametrize("newer", [
    claim("m2:c1", "Sam left OpenAI in 2024.", 1, ["Sam", "OpenAI"]),  # end of state, not an addition
    claim("m2:c1", "Sam no longer works at OpenAI.", 1, ["Sam", "OpenAI"]),
    claim("m2:c1", "Sam does not work at OpenAI.", 1, ["Sam", "OpenAI"], polarity="negative"),
    claim("m2:c1", "Sam moved to OpenAI's Paris office.", 1, ["Sam", "OpenAI"]),  # states a change
    claim("m2:c1", "Sam will join OpenAI's board.", 1, ["Sam", "OpenAI"]),  # future vs current
    claim("m2:c1", "Sam's sister works at OpenAI too.", 1, ["Sam", "OpenAI"]),  # possessive: about the sister
    claim("m2:c1", "He joined OpenAI in 2024.", 1, ["OpenAI"]),  # unresolved reference
    claim("m2:c1", "Sam joined Anthropic in 2024.", 1, ["Sam", "Anthropic"]),  # does not name OpenAI
    claim("m2:c1", "Sam works at OpenAI.", 1, ["Sam", "OpenAI"], temporal="tomorrow"),  # nothing new
])
def test_extends_is_not_created_without_safe_evidence(newer):
    _, graph = build([WORKS, newer])
    assert graph.additive == [] and graph.rejections == []


def test_extends_skips_same_slot_and_pairs_already_linked():
    a = claim("m1:c1", "Sam works at OpenAI.", 0, ["Sam", "OpenAI"], claim_type="employer")
    b = claim("m2:c1", "Sam joined OpenAI in 2024.", 1, ["Sam", "OpenAI"], claim_type="employer")
    _, graph = build([a, b])  # same slot: replacement is the versioning rules' decision, not EXTENDS
    assert graph.edges == []
    a2 = claim("m2:c1", "Sam now works at OpenAI's Paris office.", 1, ["Sam", "OpenAI"], claim_type="employer")
    _, graph = build([a, a2])
    assert edges(graph) == [("mem:m2:c1", LINK_SUPERSEDES, "mem:m1:c1")]
    _, graph = build([WORKS, JOINED], [rel("m2:r1", "m2:c1", "m1:c1", "REFERS_TO")])
    assert edges(graph) == [("mem:m2:c1", "REFERS_TO", "mem:m1:c1")]


def test_extends_requires_a_named_object_not_just_the_same_subject():
    _, graph = build([claim("m1:c1", "Sam is a designer.", 0, ["Sam"]), claim("m2:c1", "Sam is vegetarian.", 1, ["Sam"])])
    assert graph.additive == []


# ---- DERIVES ----------------------------------------------------------------


def test_no_derives_is_fabricated_without_an_explicit_inference():
    _, graph = build(BUDGET)
    assert graph.edges == [] and graph.rejections == []


def test_derives_from_supports_with_inference_cue_has_one_edge_per_parent():
    c, graph = build(BUDGET, BUDGET_SUPPORTS)
    assert edges(graph) == [("mem:m3:c1", "DERIVES", "mem:m1:c1"), ("mem:m3:c1", "DERIVES", "mem:m2:c1")]
    assert [(r.claim_id, r.relation_id, r.origin) for r in graph.additive] == [
        ("m3:c1", "m3:r1", ORIGIN_SUPPORTS), ("m3:c1", "m3:r2", ORIGIN_SUPPORTS)]
    assert all(m.status == STATUS_CURRENT for m in c.memories.values())
    assert [r.source_memory_id for r in graph.incoming("mem:m1:c1", LINK_DERIVES)] == ["mem:m3:c1"]


@pytest.mark.parametrize("relation", [
    rel("m3:r1", "m1:c1", "m3:c1", "SUPPORTS", "The Atlas budget is now €35,000."),  # confirmation/echo
    rel("m1:r1", "m1:c1", "m3:c1", "SUPPORTS", "So"),  # cue not stated with the conclusion
    rel("m3:r1", "m1:c1", "m3:c1", "CAUSED_BY", "So"),  # causal, not derivation
])
def test_derives_needs_supports_stated_as_inference_with_the_conclusion(relation):
    _, graph = build(BUDGET, [relation])
    assert graph.additive == []


def test_multiple_derives_parents_through_the_api():
    claims = [claim(f"m{i}:c1", text, i - 1) for i, text in enumerate(
        ["Ana ran 5 km on Monday.", "Ana ran 7 km on Wednesday.", "Ana ran 8 km on Friday.", "Ana ran 20 km this week."], 1)]
    _, graph = build(claims)
    rejected = graph.derive("mem:m4:c1", ["mem:m1:c1", "mem:m2:c1", "mem:m3:c1"], "m4:c1", "sum of the week's runs")
    assert rejected == []
    assert edges(graph) == [("mem:m4:c1", "DERIVES", f"mem:m{i}:c1") for i in (1, 2, 3)]


# ---- existing relationships --------------------------------------------------


def test_update_supersede_contradict_and_refers_to_links_are_carried_unchanged():
    claims = [claim("m1:c1", "The user lives in Lisbon.", 0, ["Lisbon"], claim_type="residence"),
              claim("m2:c1", "The user now lives in Porto.", 1, ["Porto"], claim_type="residence"),
              claim("m3:c1", "The meeting is on Monday.", 2, ["meeting"]),
              claim("m4:c1", "The meeting is on Tuesday.", 3, ["meeting"]),
              claim("m5:c1", "The user has a cat called Miso.", 4, ["Miso"]),
              claim("m6:c1", "Miso is ten years old.", 5, ["Miso"]),
              claim("m7:c1", "The user moved to Braga.", 6, ["Braga"], claim_type="residence")]
    relations = [rel("m4:r1", "m4:c1", "m3:c1", "CONTRADICTS"), rel("m6:r1", "m6:c1", "m5:c1", "REFERS_TO"),
                 rel("m7:r1", "m7:c1", "m2:c1", "UPDATES")]
    c, graph = build(claims, relations)
    assert [(link.newer_memory_id, link.type, link.older_memory_id, link.claim_id, link.relation_id) for link in c.links] == [
        (r.source_memory_id, r.type, r.target_memory_id, r.claim_id, r.relation_id) for r in graph.links]
    assert {r.type for r in graph.links} == {LINK_SUPERSEDES, LINK_CONTRADICTS, LINK_REFERS_TO, LINK_UPDATES}
    assert all(r.origin == ORIGIN_CONSOLIDATION and r.reason for r in graph.links)
    status = {m.content: m.status for m in c.memories.values()}
    assert status["The user lives in Lisbon."] == status["The user now lives in Porto."] == STATUS_SUPERSEDED
    assert status["The meeting is on Monday."] == status["The meeting is on Tuesday."] == STATUS_CONTESTED
    assert status["The user moved to Braga."] == STATUS_CURRENT
    assert versioning_cycles(c.links) == []


def test_versioning_cycles_detects_mutual_updates():
    links = [MemoryLink("a", "b", LINK_UPDATES, "x"), MemoryLink("b", "a", LINK_SUPERSEDES, "y")]
    assert versioning_cycles(links) == [("a", "b")]


# ---- validation ---------------------------------------------------------------


def _graph():
    return build([WORKS, JOINED, claim("m3:c1", "Sam lives in Porto.", 2, ["Sam", "Porto"])])[1]


def edge(source, target, type_=LINK_DERIVES, conv=CONV, origin="explicit"):
    return MemoryRelationship(conv, source, target, type_, "m3:c1", None, origin, "test")


@pytest.mark.parametrize("relationship, code", [
    (edge("mem:m3:c1", "mem:m3:c1"), SELF_LINK),
    (edge("mem:m3:c1", "mem:m1:c1", conv="other"), CROSS_CONVERSATION),
    (edge("mem:m9:c1", "mem:m1:c1"), MISSING_SOURCE),
    (edge("mem:m3:c1", "mem:m9:c1"), MISSING_TARGET),
    (edge("mem:m3:c1", "mem:m1:c1", LINK_UPDATES), INVALID_TYPE),
    (edge("mem:m3:c1", "mem:m1:c1", "IMPLIES"), INVALID_TYPE),
    (edge("mem:m3:c1", "mem:m1:c1", origin=ORIGIN_CONSOLIDATION), INVALID_ORIGIN),
    (edge("mem:m1:c1", "mem:m2:c1", LINK_EXTENDS), ALREADY_RELATED),  # reverse of the rule's edge
    (edge("mem:m2:c1", "mem:m1:c1"), ALREADY_RELATED),  # DERIVES on an EXTENDS pair
])
def test_invalid_relationships_are_rejected_with_a_code(relationship, code):
    graph = _graph()
    before = graph.edges
    rejection = graph.add(relationship)
    assert rejection is not None and rejection.code == code and graph.rejections == [rejection]
    assert graph.edges == before


def test_duplicates_and_cycles_are_rejected():
    graph = _graph()
    assert graph.add(edge("mem:m3:c1", "mem:m1:c1")) is None
    assert graph.add(edge("mem:m3:c1", "mem:m1:c1")).code == DUPLICATE
    graph2 = build([claim(f"m{i}:c1", f"Fact {i} about Kai.", i, ["Kai"]) for i in range(1, 4)])[1]
    assert graph2.add(edge("mem:m1:c1", "mem:m2:c1")) is None
    assert graph2.add(edge("mem:m2:c1", "mem:m3:c1")) is None
    assert graph2.add(edge("mem:m3:c1", "mem:m1:c1")).code == CYCLE


def test_memory_from_another_conversation_is_rejected():
    graph = _graph()
    foreign = replace(graph.memories["mem:m1:c1"], conversation_id="other")
    graph.memories["mem:m1:c1"] = foreign
    assert graph.add(edge("mem:m3:c1", "mem:m1:c1")).code == CROSS_CONVERSATION


def test_identical_memories_in_two_conversations_never_link():
    graphs = {conv: build([replace(WORKS, provenance=replace(WORKS.provenance, conversation_id=conv)),
                           replace(JOINED, provenance=replace(JOINED.provenance, conversation_id=conv))], conv=conv)[1]
              for conv in ("a", "b")}
    for conv, graph in graphs.items():
        assert {r.conversation_id for r in graph.edges} == {conv}
    a_edge = graphs["a"].additive[0]
    assert graphs["b"].add(a_edge).code == CROSS_CONVERSATION
    assert a_edge.relationship_id != graphs["b"].additive[0].relationship_id


def test_relationship_ids_are_deterministic_and_unambiguous():
    assert relationship_id("a/b", "mem:m1:c1", "EXTENDS", "mem:m0:c1") == "rel:a%2Fb/mem:m1:c1/EXTENDS/mem:m0:c1"
    assert relationship_id("a/b", "c", "DERIVES", "d") != relationship_id("a", "b/c", "DERIVES", "d")


# ---- persistence ----------------------------------------------------------------


class ScriptedClient:
    """Scripted atomizer: claims (content, entities[, claim_type]) and relations per target message."""

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


def conversation(cid, *texts):
    return parse_conversation({"conversation_id": cid, "messages": [
        {"id": f"m{i}", "role": "user", "content": text} for i, text in enumerate(texts, 1)]})


PROJECT = conversation("atlas", "Sam works at OpenAI.", "Sam joined OpenAI in 2024.", "The Atlas budget was €20,000.",
                       "The Atlas budget increased by €15,000.", "So the Atlas budget is now €35,000.")
PROJECT_CLAIMS = {"m1": [("Sam works at OpenAI.", ["Sam", "OpenAI"])],
                  "m2": [("Sam joined OpenAI in 2024.", ["Sam", "OpenAI"])],
                  "m3": [("The Atlas budget was €20,000.", ["Atlas"])],
                  "m4": [("The Atlas budget increased by €15,000.", ["Atlas"])],
                  "m5": [("The Atlas budget is now €35,000.", ["Atlas"])]}
PROJECT_RELATIONS = {"m5": [("m3:c1", "m5:c1", "SUPPORTS", "So"), ("m4:c1", "m5:c1", "SUPPORTS", "So")]}


def ingest(store, conv=PROJECT, claims=PROJECT_CLAIMS, relations=PROJECT_RELATIONS):
    report = V2IngestionPipeline(store, SemanticAtomizer(ScriptedClient(claims, relations)), targets_per_window=2,
                                 context_before=2).ingest(conv)
    assert not report.issues, report.issues
    consolidate_stored(store.connection, conv.conversation_id)
    return build_relationships_stored(store.connection, conv.conversation_id)


def rows(store):
    return store.connection.execute("SELECT * FROM v2_memory_relationships ORDER BY conversation_id, ordinal").fetchall()


EXPECTED = [("mem:m2:c1", "EXTENDS", "mem:m1:c1"), ("mem:m5:c1", "DERIVES", "mem:m3:c1"),
            ("mem:m5:c1", "DERIVES", "mem:m4:c1")]


def test_stored_graph_round_trips_and_links_table_is_untouched():
    store = Store(":memory:")
    graph = ingest(store)
    assert edges(graph) == EXPECTED
    links_before = MemoryStore(store.connection).get_links("atlas")
    stored = RelationshipStore(store.connection).get("atlas")
    assert [r.to_dict() for r in stored.edges] == [r.to_dict() for r in graph.edges]
    assert MemoryStore(store.connection).get_links("atlas") == links_before
    assert store.connection.execute("SELECT COUNT(*) FROM v2_memory_versions").fetchone()[0] == len(links_before)


def test_reingestion_recreates_exactly_one_copy_of_each_edge():
    store = Store(":memory:")
    ingest(store)
    first = rows(store)
    ingest(store)
    build_relationships_stored(store.connection, "atlas")
    assert rows(store) == first and len(first) == len(EXPECTED)


def test_schema_enforces_uniqueness_type_and_no_self_links():
    store = Store(":memory:")
    ingest(store)
    insert = ("INSERT INTO v2_memory_relationships (conversation_id, source_memory_id, target_memory_id, type, ordinal, "
              "claim_id, relation_id, origin, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    for values in (("atlas", "mem:m2:c1", "mem:m1:c1", "EXTENDS", 99, "m2:c1", None, "extends_rule", "x"),
                   ("atlas", "mem:m1:c1", "mem:m1:c1", "DERIVES", 99, "m1:c1", None, "explicit", "x"),
                   ("atlas", "mem:m3:c1", "mem:m1:c1", "UPDATES", 99, "m3:c1", None, "explicit", "x"),
                   ("other", "mem:m3:c1", "mem:m1:c1", "DERIVES", 99, "m3:c1", None, "explicit", "x")):
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(insert, values)


def test_reconsolidation_and_v1_conversation_save_cascade_edges_away():
    store = Store(":memory:")
    ingest(store)
    consolidate_stored(store.connection, "atlas")  # memories replaced; edges must be rebuilt
    assert rows(store) == []
    build_relationships_stored(store.connection, "atlas")
    assert len(rows(store)) == len(EXPECTED)
    store.save_conversation(PROJECT, [], np.zeros((0, 0), dtype=np.float32), "none")
    assert rows(store) == []


def test_conversations_are_isolated_in_storage():
    store = Store(":memory:")
    ingest(store)
    other = conversation("atlas-2", *[m.content for m in PROJECT.messages])
    graph = ingest(store, other)
    assert edges(graph) == EXPECTED and {r[0] for r in rows(store)} == {"atlas", "atlas-2"}
    assert all(r.conversation_id == "atlas-2" for r in RelationshipStore(store.connection).get("atlas-2").edges)


def test_two_independent_runs_are_byte_identical():
    def run():
        store = Store(":memory:")
        graph = ingest(store)
        return json.dumps([graph.to_dict(), rows(store)], sort_keys=True)

    assert run() == run()


def test_stored_graph_matches_in_memory_graph():
    store = Store(":memory:")
    stored = ingest(store)
    semantic = SemanticStore(store.connection)
    claims, relations = semantic.get_claims("atlas"), semantic.get_relations("atlas")
    c = consolidate("atlas", claims, relations)
    memory = build_relationships("atlas", c.memories.values(), c.links, c.sources, c.decisions, claims, relations)
    assert [r.to_dict() for r in memory.edges] == [r.to_dict() for r in stored.edges]
