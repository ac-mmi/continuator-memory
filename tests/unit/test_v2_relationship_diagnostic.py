import ast
import inspect
import json

import pytest

from continuator_memory.store import Store
from continuator_memory.v2 import relationship_diagnostic as rd
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.expansion import INCOMING, OUTGOING, expand
from continuator_memory.v2.expansion_store import build_knowledge_stored
from continuator_memory.v2.memory import STATUS_CURRENT, STATUS_SUPERSEDED
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.relationship_diagnostic import (
    ACCEPTED_EXISTING,
    DUPLICATE,
    INSUFFICIENT_EVIDENCE,
    REJECTED_RULE,
    connections,
    decide,
    generate_candidates,
    load_evidence,
    oracle_knowledge,
    traversal,
)
from tests.unit.test_v2_expansion import ScriptedClient, Seed, conversation, user

CASES = {
    "atlas": (user("The Atlas budget was €20,000.", "The Atlas budget increased by €15,000.",
                   "So the Atlas budget is now €35,000."),
              {"m1": [("The Atlas budget was €20,000.", ["Atlas"])],
               "m2": [("The Atlas budget increased by €15,000.", ["Atlas"])],
               "m3": [("The Atlas budget is now €35,000.", ["Atlas"])]},
              {"m3": [("m1:c1", "m3:c1", "SUPPORTS", "So"), ("m2:c1", "m3:c1", "SUPPORTS", "So")]}),
    "hana": (user("Hana will lead the team from May 1st.", "Hana takes over as team lead starting May 1st."),
             {"m1": [("Hana will lead the team from May 1st.", ["Hana", "team"])],
              "m2": [("Hana takes over as team lead starting May 1st.", ["Hana", "team"])]},
             {"m2": [("m1:c1", "m2:c1", "SUPPORTS", "Hana")]}),
    "sam": (user("Sam works at OpenAI.", "Sam joined OpenAI in 2024."),
            {"m1": [("Sam works at OpenAI.", ["Sam", "OpenAI"])], "m2": [("Sam joined OpenAI in 2024.", ["Sam", "OpenAI"])]},
            {}),
    "update": (user("I live in Lisbon.", "Actually, I moved to Braga."),
               {"m1": [("The user lives in Lisbon.", ["Lisbon"], "residence")],
                "m2": [("The user moved to Braga.", ["Braga"], "residence")]},
               {"m2": [("m2:c1", "m1:c1", "UPDATES", "Actually")]}),
    "mei": (user("Mei is our treasurer.", "The budget will be handled by Mei.", "Mei prefers tea."),
            {"m1": [("Mei is the treasurer.", ["Mei"])], "m2": [("The budget will be handled by Mei.", ["budget"])],
             "m3": [("Mei prefers tea.", ["tea"])]}, {}),
    "omar": (user("Omar gets seasick, so no boat trips for Omar."),
             {"m1": [("Omar gets seasick.", ["Omar"]), ("There will be no boat trips for Omar.", ["Omar"])]}, {}),
}


@pytest.fixture(scope="module")
def corpus():
    store = Store(":memory:")
    evidence = {}
    for cid, (messages, claims, relations) in CASES.items():
        report = V2IngestionPipeline(store, SemanticAtomizer(ScriptedClient(claims, relations)), targets_per_window=2,
                                     context_before=2).ingest(conversation(cid, *messages))
        assert not report.issues, report.issues
        evidence[cid] = load_evidence(store.connection, cid, build_knowledge_stored(store.connection, cid))
    return store, evidence


def found(evidence, cid):
    return [(c.source_memory_id, c.relationship_type, c.target_memory_id, c.decision, c.candidate_reason,
             c.failing_checks) for c in generate_candidates(evidence[cid])]


# ---- candidate generation -------------------------------------------------------------------


def test_existing_edges_are_accepted_and_repeated_explicit_relations_are_duplicates(corpus):
    _, evidence = corpus
    assert found(evidence, "update") == [
        ("mem:m2:c1", "UPDATES", "mem:m1:c1", ACCEPTED_EXISTING, rd.EXISTING_VERSION_CHAIN, ()),
        ("mem:m2:c1", "UPDATES", "mem:m1:c1", DUPLICATE, rd.EXPLICIT_RELATION, ())]
    assert found(evidence, "sam") == [("mem:m2:c1", "EXTENDS", "mem:m1:c1", ACCEPTED_EXISTING, rd.EXISTING_LINK, ())]
    atlas = found(evidence, "atlas")
    assert atlas[:4] == [
        ("mem:m3:c1", "DERIVES", "mem:m1:c1", ACCEPTED_EXISTING, rd.EXISTING_LINK, ()),
        ("mem:m3:c1", "DERIVES", "mem:m2:c1", ACCEPTED_EXISTING, rd.EXISTING_LINK, ()),
        ("mem:m3:c1", "DERIVES", "mem:m1:c1", DUPLICATE, rd.SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE, ()),
        ("mem:m3:c1", "DERIVES", "mem:m2:c1", DUPLICATE, rd.SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE, ())]


def test_every_existing_edge_is_reported_once_as_accepted(corpus):
    _, evidence = corpus
    for cid, ev in evidence.items():
        accepted = {(c.source_memory_id, c.relationship_type, c.target_memory_id)
                    for c in generate_candidates(ev) if c.decision == ACCEPTED_EXISTING}
        edges = {(r.source_memory_id, r.type, r.target_memory_id) for r in [*ev.graph.links, *ev.graph.additive]}
        assert accepted == edges, cid


def test_candidates_are_deterministic_unique_and_never_all_pairs(corpus):
    _, evidence = corpus
    for ev in evidence.values():
        first, second = generate_candidates(ev), generate_candidates(ev)
        assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
        live = [c.key for c in first if c.decision != DUPLICATE]
        assert len(live) == len(set(live))
        assert all(c.candidate_reason in rd.CANDIDATE_REASONS and c.evidence for c in first)
        assert all(c.anchor in {rd.ANCHOR_SUBJECT, rd.ANCHOR_ENTITY, rd.ANCHOR_TEXT, rd.ANCHOR_RELATION,
                                rd.ANCHOR_MESSAGE, rd.ANCHOR_EDGE} for c in first)


# ---- decisions ------------------------------------------------------------------------------


def test_decision_follows_the_reason_classes():
    assert decide(["NO_OTHER_ENTITY"]) == REJECTED_RULE
    assert decide(["NO_OTHER_ENTITY", "MISSING_ENTITY"]) == INSUFFICIENT_EVIDENCE
    assert decide(["MISSING_ENTITY", "DIFFERENT_SUBJECT"]) == rd.NOT_APPLICABLE
    assert {c for c, _ in rd.REASONS.values()} == {rd.NA, rd.GAP, rd.RULE}


def test_supports_without_an_inference_cue_is_rejected_by_the_derives_rule(corpus):
    _, evidence = corpus
    assert found(evidence, "hana") == [("mem:m2:c1", "DERIVES", "mem:m1:c1", REJECTED_RULE,
                                        rd.SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE, ("NO_EXPLICIT_INFERENCE_CUE",))]


def test_rule_rejection_records_the_first_failing_extends_check(corpus):
    _, evidence = corpus
    [c] = [c for c in generate_candidates(evidence["atlas"]) if c.relationship_type == "EXTENDS"]
    assert (c.decision, c.anchor, c.rejection_reason, c.failing_checks) == (
        REJECTED_RULE, "subject_field", "NO_OTHER_ENTITY", ("NO_OTHER_ENTITY", "TEMPORAL_CLASS"))
    for ev in evidence.values():
        for c in generate_candidates(ev):
            if c.relationship_type == "EXTENDS" and c.missing:
                assert rd.extends_checks(ev, c.target_memory_id, c.source_memory_id, c.anchor) == list(c.failing_checks)


def test_a_name_only_shared_in_text_is_insufficient_evidence(corpus):
    _, evidence = corpus
    mei = generate_candidates(evidence["mei"])
    assert [(c.source_memory_id, c.target_memory_id) for c in mei] == [
        ("mem:m2:c1", "mem:m1:c1"), ("mem:m3:c1", "mem:m1:c1"), ("mem:m3:c1", "mem:m2:c1")]
    assert all(c.decision == INSUFFICIENT_EVIDENCE and c.anchor == "text_only" and c.rejection_reason == "MISSING_ENTITY"
               for c in mei)


def test_an_inference_cue_without_supports_is_insufficient_evidence(corpus):
    _, evidence = corpus
    derives = [c for c in generate_candidates(evidence["omar"]) if c.relationship_type == "DERIVES"]
    assert [(c.source_memory_id, c.target_memory_id, c.decision, c.candidate_reason, c.failing_checks)
            for c in derives] == [("mem:m1:c2", "mem:m1:c1", INSUFFICIENT_EVIDENCE, rd.INFERENCE_CUE_WITHOUT_RELATION,
                                   ("NO_SUPPORTING_RELATION",))]


# ---- phase 5D compatibility -----------------------------------------------------------------


def test_traversal_mirrors_the_phase5d_policy():
    def t(type_, seed, status=STATUS_CURRENT, past=False):
        return traversal("a", "b", type_, seed, status, past)

    assert t("EXTENDS", "a").traversable and t("EXTENDS", "a").direction == OUTGOING
    assert t("EXTENDS", "b").traversable and t("EXTENDS", "b").direction == INCOMING
    assert t("REFERS_TO", "a").traversable and not t("REFERS_TO", "b").traversable
    assert t("UPDATES", "b").traversable
    assert t("UPDATES", "a").traversable and not t("UPDATES", "a", STATUS_SUPERSEDED).traversable
    assert t("UPDATES", "a", STATUS_SUPERSEDED, past=True).traversable
    assert t("DERIVES", "a").role == "support" and t("DERIVES", "b").role == "conclusion"


def test_connections_check_both_directions_and_skip_duplicates(corpus):
    _, evidence = corpus
    mei = generate_candidates(evidence["mei"])
    statuses = {m: x.status for m, x in evidence["mei"].graph.memories.items()}
    [out] = connections(mei, ["mem:m3:c1"], ["mem:m1:c1"], statuses, False)
    [inc] = connections(mei, ["mem:m1:c1"], ["mem:m3:c1"], statuses, False)
    assert (out.traversal.direction, inc.traversal.direction) == (OUTGOING, INCOMING)
    assert out.candidate is inc.candidate and out.traversal.traversable and inc.traversal.traversable
    assert connections(mei, ["mem:m1:c1"], ["mem:m1:c1"], statuses, False) == []
    atlas = generate_candidates(evidence["atlas"])
    statuses = {m: x.status for m, x in evidence["atlas"].graph.memories.items()}
    [c] = connections(atlas, ["mem:m3:c1"], ["mem:m1:c1"], statuses, False)
    assert c.candidate.decision == ACCEPTED_EXISTING


# ---- oracle ---------------------------------------------------------------------------------


def test_oracle_graph_is_an_isolated_copy(corpus):
    _, evidence = corpus
    ev = evidence["mei"]
    before = json.dumps(ev.graph.to_dict(), sort_keys=True)
    chosen = [c for c in generate_candidates(ev) if c.source_memory_id == "mem:m3:c1"]
    result = oracle_knowledge(ev.knowledge, chosen)
    assert json.dumps(ev.graph.to_dict(), sort_keys=True) == before
    assert result.knowledge.graph is not ev.graph and result.added == [c.candidate_id for c in chosen]
    assert all(r.reason == "oracle relationship diagnostic" for r in result.knowledge.graph.additive)
    seeds = [Seed(1, "mem:m3:c1", "mei", 1.0)]
    real = expand("What does Mei prefer?", seeds, {"mei": ev.knowledge})
    oracle = expand("What does Mei prefer?", seeds, {"mei": result.knowledge})
    assert [i.memory_id for i in real.expanded] == []
    assert sorted(i.memory_id for i in oracle.expanded) == ["mem:m1:c1", "mem:m2:c1"]


def test_oracle_adds_only_missing_candidates_and_keeps_existing_edges(corpus):
    _, evidence = corpus
    ev = evidence["update"]
    result = oracle_knowledge(ev.knowledge, generate_candidates(ev))
    assert result.added == [] and len(result.knowledge.graph.links) == len(ev.graph.links)


# ---- no leakage -----------------------------------------------------------------------------


def test_candidate_generation_never_sees_a_question_or_gold_label():
    assert list(inspect.signature(generate_candidates).parameters) == ["ev"]
    names = {n.id for n in ast.walk(ast.parse(inspect.getsource(rd))) if isinstance(n, ast.Name)}
    names |= {n.arg for n in ast.walk(ast.parse(inspect.getsource(rd))) if isinstance(n, ast.arg)}
    assert not {n for n in names if any(w in n.lower() for w in ("question", "gold", "expected", "answer"))}
