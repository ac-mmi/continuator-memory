"""Structured extraction: contract, validation, attribution, storage and isolation, with a scripted model."""

from __future__ import annotations

import json

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import AtomizationRequestError, V2Settings
from continuator_memory.v2.models import SUPPORT_IDENTITY, SUPPORT_SOURCE
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.structured_extraction import (
    STRUCTURED_EXAMPLES,
    STRUCTURED_SYSTEM_PROMPT,
    SUPPORT_REVIEW,
    StructuredExtractor,
    structured_extractor,
    structured_output_schema,
    validate_structured,
)
from continuator_memory.v2.structured_store import StructuredStore, ingest_structured
from continuator_memory.v2.windowing import ContextWindow, build_windows
from continuator_memory.webapi.extract import EXTRACT_MODES

CID = "c1"
TS = "2024-05-10"


def msg(mid, role, text, position=0, ts=TS):
    return Message(mid, CID, position, role, text, ts)


def window(*targets, context=()):
    return ContextWindow(CID, 0, tuple(context), tuple(targets))


def ev(mid, quote):
    return {"message_id": mid, "quote": quote}


def memory(statement, subject, evidence, category="fact", assertion="assertion", predicate="", obj="", reason="",
           expression="", normalized="", kind="unspecified"):
    return {"statement": statement, "subject": subject, "predicate": predicate, "object": obj, "category": category,
            "assertion_type": assertion, "reason": reason,
            "temporal": {"expression": expression, "normalized": normalized, "kind": kind}, "evidence": evidence}


def relationship(subject, rtype, obj, evidence, assertion="assertion"):
    return {"subject": subject, "type": rtype, "object": obj, "assertion_type": assertion,
            "temporal": {"expression": "", "normalized": "", "kind": "unspecified"}, "evidence": evidence}


def entity(name, etype, message_ids, aliases=()):
    return {"name": name, "type": etype, "aliases": list(aliases), "message_ids": message_ids}


def run(w, entities=(), memories=(), relationships=(), prior=()):
    raw = json.dumps({"entities": list(entities), "memories": list(memories), "relationships": list(relationships)})
    return validate_structured(raw, w, prior)


def codes(result):
    return [i.code for i in result.issues]


NADIA = msg("m1", "user", "Nadia: I'm building Bluebird with Omar at Finch Labs. I prefer written notes.", 0)
OMAR = msg("m2", "assistant", "Omar: I live in Quito, and I started on 3 March 2024.", 1)


def test_multiple_entities_in_one_window_keep_type_message_and_speaker():
    result = run(window(NADIA, OMAR), entities=[
        entity("Nadia", "person", ["m1"]), entity("Bluebird", "project", ["m1"]),
        entity("Finch Labs", "company", ["m1"]), entity("Quito", "place", ["m2"]), entity("Omar", "person", ["m2"])])
    assert [(e.name, e.type) for e in result.entities] == [
        ("Nadia", "person"), ("Bluebird", "project"), ("Finch Labs", "company"), ("Quito", "place"), ("Omar", "person")]
    assert result.entities[3].message_ids == ("m2",)
    assert result.entities[3].speakers == ("Omar",)
    assert result.entities[0].entity_id == "c1:w0000:e1"


def test_entity_names_are_not_resolved_across_mentions():
    a = msg("a1", "user", "Nadia: Bluebird shipped. Bird is my nickname for it.")
    result = run(window(a), entities=[entity("Bluebird", "project", ["a1"]), entity("Bird", "project", ["a1"])])
    assert [e.name for e in result.entities] == ["Bluebird", "Bird"]


def test_supported_alias_kept_and_unsupported_alias_dropped():
    a = msg("a1", "user", "Nadia: Bluebird, also called Bird, shipped.")
    result = run(window(a), entities=[entity("Bluebird", "project", ["a1"], aliases=["Bird", "Sparrow"])])
    assert result.entities[0].aliases == ("Bird",)
    assert "UNSUPPORTED_ALIAS" in codes(result)


def test_new_entity_type_is_kept_and_flagged():
    a = msg("a1", "user", "Nadia: Bluebird uses Rust.")
    result = run(window(a), entities=[entity("Rust", "language", ["a1"])])
    assert result.entities[0].type == "language"
    assert "UNKNOWN_ENTITY_TYPE" in codes(result)


def test_multiple_independent_facts_from_one_message_get_distinct_ids():
    result = run(window(NADIA), memories=[
        memory("Nadia is building Bluebird.", "Nadia", [ev("m1", "I'm building Bluebird")], obj="Bluebird"),
        memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")], "preference")])
    assert [m.memory_id for m in result.memories] == ["c1:w0000:m1", "c1:w0000:m2"]
    assert [m.category for m in result.memories] == ["fact", "preference"]
    assert all(m.support == SUPPORT_SOURCE and m.confirmed for m in result.memories)


def test_structured_fields_are_preserved():
    result = run(window(NADIA), memories=[memory("Nadia is building Bluebird.", "Nadia", [ev("m1", "I'm building Bluebird")],
                                                  predicate="is building", obj="Bluebird")])
    m = result.memories[0]
    assert (m.subject, m.predicate, m.object) == ("Nadia", "is building", "Bluebird")
    assert m.message_ids == ("m1",)


def test_relationships_between_entities_and_open_vocabulary():
    result = run(window(NADIA), relationships=[
        relationship("Nadia", "develops", "Bluebird", [ev("m1", "I'm building Bluebird")]),
        relationship("Nadia", "Collaborates With", "Omar", [ev("m1", "I'm building Bluebird with Omar")])])
    assert [(r.subject, r.type, r.object) for r in result.relationships] == [
        ("Nadia", "develops", "Bluebird"), ("Nadia", "collaborates_with", "Omar")]
    assert result.relationships[0].relationship_id == "c1:w0000:r1"


def test_rename_relationship_resolves_a_pronoun_from_the_window():
    earlier = msg("a1", "user", "Omar: How is Bluebird?", 0)
    rename = msg("a2", "user", "Nadia: I'm renaming it Songbird.", 1)
    result = run(window(rename, context=[earlier]), relationships=[
        relationship("Songbird", "formerly_named", "Bluebird", [ev("a2", "I'm renaming it Songbird")])])
    assert result.relationships[0].type == "formerly_named"


def test_co_occurrence_does_not_create_a_relationship():
    w = window(NADIA, OMAR)
    result = run(w, relationships=[
        relationship("Omar", "lives_with", "Bluebird", [ev("m2", "I live in Quito")]),
        relationship("Bluebird", "located_in", "Quito", [ev("m1", "I'm building Bluebird")])])
    assert result.relationships == []
    assert codes(result).count("ENDPOINT_NOT_IN_EVIDENCE") == 2


def test_relationship_without_evidence_is_rejected():
    result = run(window(NADIA), relationships=[relationship("Nadia", "develops", "Bluebird", [])])
    assert result.relationships == []
    assert codes(result) == ["MISSING_EVIDENCE"]


def test_speaker_comes_from_the_message_not_the_model():
    result = run(window(OMAR), memories=[memory("Omar lives in Quito.", "Omar", [ev("m2", "I live in Quito")])])
    m = result.memories[0]
    assert (m.speaker, m.speaker_role, m.reported, m.confirmed) == ("Omar", "assistant", False, True)


def test_statement_about_another_speaker_stays_reported():
    other = msg("a1", "user", "Nadia: Thanks.", 0)
    says = msg("a2", "assistant", "Omar: Nadia lives in Lima, and she is moving soon.", 1)
    result = run(window(says, context=[other]), memories=[memory("Nadia lives in Lima.", "Nadia", [ev("a2", "Nadia lives in Lima")])])
    m = result.memories[0]
    assert (m.speaker, m.reported, m.confirmed) == ("Omar", True, False)


def test_explicit_confirmation_is_attributed_to_the_confirming_speaker():
    question = msg("a1", "assistant", "Omar: Are you still in Lima?", 0)
    reply = msg("a2", "user", "Nadia: Yes, for now.", 1)
    result = run(window(reply, context=[question]), memories=[memory(
        "Nadia is still in Lima.", "Nadia", [ev("a1", "Are you still in Lima?"), ev("a2", "Yes")], assertion="confirmation")])
    m = result.memories[0]
    assert (m.speaker, m.assertion_type, m.reported, m.confirmed) == ("Nadia", "confirmation", False, True)
    assert m.message_ids == ("a1", "a2")


def test_confirmation_without_the_other_speaker_needs_review():
    result = run(window(NADIA), memories=[memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")],
                                                  assertion="confirmation")])
    assert result.memories[0].review_reasons == ("CONFIRMATION_WITHOUT_PROMPT",)


@pytest.mark.parametrize("assertion", ["question", "hypothetical", "suggestion", "assumption"])
def test_non_assertions_are_kept_but_never_confirmed(assertion):
    a = msg("a1", "assistant", "Omar: You could try using Rust for Bluebird?")
    result = run(window(a), memories=[memory("Omar could try using Rust for Bluebird.", "Omar",
                                             [ev("a1", "You could try using Rust for Bluebird")], assertion=assertion)])
    assert result.memories[0].assertion_type == assertion
    assert not result.memories[0].confirmed


def test_question_labelled_as_a_fact_needs_review():
    a = msg("a1", "assistant", "Omar: Do you live in Lima?")
    result = run(window(a), memories=[memory("Omar lives in Lima.", "Omar", [ev("a1", "Do you live in Lima?")])])
    assert "QUESTION_AS_FACT" in result.memories[0].review_reasons
    assert not result.memories[0].confirmed


def test_decision_keeps_its_reason():
    a = msg("a1", "user", "Nadia: I dropped Postgres because hosting costs too much.")
    result = run(window(a), memories=[memory("Nadia dropped Postgres.", "Nadia", [ev("a1", "I dropped Postgres because hosting costs too much")],
                                             "decision", reason="hosting costs too much")])
    assert result.memories[0].category == "decision"
    assert result.memories[0].reason == "hosting costs too much"


def test_reason_with_no_source_is_dropped_without_losing_the_decision():
    a = msg("a1", "user", "Nadia: I dropped Postgres.")
    result = run(window(a), memories=[memory("Nadia dropped Postgres.", "Nadia", [ev("a1", "I dropped Postgres")], "decision",
                                             reason="licensing fees")])
    assert result.memories[0].reason is None
    assert "REASON_NOT_IN_SOURCE" in codes(result)


def test_historical_and_future_dates_keep_expression_and_normalized_value():
    a = msg("a1", "user", "Nadia: I joined Finch in March 2023, and Bluebird launches on 3 June.", ts="2024-05-10")
    result = run(window(a), memories=[
        memory("Nadia joined Finch in March 2023.", "Nadia", [ev("a1", "I joined Finch in March 2023")], "event",
               expression="in March 2023", normalized="2023-03", kind="historical"),
        memory("Bluebird launches on 3 June.", "Bluebird", [ev("a1", "Bluebird launches on 3 June")], "plan",
               expression="on 3 June", normalized="2024-06-03", kind="future")])
    assert [(m.temporal.expression, m.temporal.normalized, m.temporal.kind) for m in result.memories] == [
        ("in March 2023", "2023-03", "historical"), ("on 3 June", "2024-06-03", "future")]


def test_normalized_date_needs_a_timestamp_when_the_year_is_not_stated():
    a = msg("a1", "user", "Nadia: Bluebird launches on 3 June.", ts=None)
    result = run(window(a), memories=[memory("Bluebird launches on 3 June.", "Bluebird", [ev("a1", "Bluebird launches on 3 June")],
                                             expression="on 3 June", normalized="2024-06-03", kind="future")])
    t = result.memories[0].temporal
    assert (t.expression, t.normalized) == ("on 3 June", None)
    assert "RESOLVED_WITHOUT_TIMESTAMP" in codes(result)


@pytest.mark.parametrize(("expression", "normalized", "kind", "code"), [
    ("on 3 June", "2024-07-03", "future", "TEMPORAL_MISMATCH"),
    ("on 3 June", "2024-06-04", "future", "TEMPORAL_MISMATCH"),
    ("on 3 June", "2024-13-40", "future", "MALFORMED_TEMPORAL"),
    ("on 3 June", "next month", "future", "MALFORMED_TEMPORAL"),
    ("on 3 June", "2023-06-03", "future", "TEMPORAL_IMPLAUSIBLE"),
])
def test_invented_or_impossible_dates_are_dropped(expression, normalized, kind, code):
    a = msg("a1", "user", "Nadia: Bluebird launches on 3 June.")
    result = run(window(a), memories=[memory("Bluebird launches on 3 June.", "Bluebird", [ev("a1", "Bluebird launches on 3 June")],
                                             expression=expression, normalized=normalized, kind=kind)])
    assert result.memories[0].temporal.normalized is None
    assert code in codes(result)


def test_temporal_expression_not_in_the_message_is_dropped():
    result = run(window(NADIA), memories=[memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")],
                                                  expression="last year", kind="historical")])
    assert result.memories[0].temporal.expression is None
    assert "TEMPORAL_NOT_IN_SOURCE" in codes(result)


def test_correction_and_condition_are_labels_not_global_state():
    a = msg("a1", "user", "Nadia: Actually I moved to Lima. I might move again if work changes.")
    result = run(window(a), memories=[
        memory("Nadia moved to Lima.", "Nadia", [ev("a1", "Actually I moved to Lima")], "event", kind="correction"),
        memory("Nadia might move again if work changes.", "Nadia", [ev("a1", "I might move again if work changes")],
               "condition", "hypothetical", kind="conditional")])
    assert [m.temporal.kind for m in result.memories] == ["correction", "conditional"]
    for record in result.memories:
        data = record.to_dict()
        assert not {"status", "superseded", "superseded_by", "is_current", "latest"} & set(data)


def test_contract_has_no_global_state_or_model_ids():
    schema = json.dumps(structured_output_schema(window(NADIA)))
    for word in ("superseded", "is_current", "claim_id", "memory_id", "entity_id"):
        assert word not in schema


def test_harmless_wording_difference_is_kept_for_review():
    a = msg("a1", "user", "Nadia: Please send me written checklists.")
    result = run(window(a), memories=[memory("Nadia prefers receiving written checklists.", "Nadia",
                                             [ev("a1", "send me written checklists")], "preference")])
    m = result.memories[0]
    assert (m.support, m.review_reasons, m.confirmed) == (SUPPORT_REVIEW, ("LEXICAL_DIFFERENCE",), False)


def test_name_from_earlier_in_the_window_resolves_identity():
    earlier = msg("a1", "assistant", "Omar: How is Bluebird?", 0)
    reply = msg("a2", "user", "Nadia: I'm keeping it deliberately small.", 1)
    result = run(window(reply, context=[earlier]), memories=[memory(
        "Nadia is keeping Bluebird deliberately small.", "Nadia", [ev("a2", "I'm keeping it deliberately small")])])
    assert result.memories[0].support == SUPPORT_IDENTITY
    assert result.memories[0].message_ids == ("a2",)


def test_answer_citing_its_question_is_fully_supported():
    question = msg("a1", "user", "Nadia: Are you still doing the repair clinic?", 0)
    reply = msg("a2", "assistant", "Omar: Yes, twice a month.", 1)
    result = run(window(reply, context=[question]), memories=[memory(
        "Omar is doing the repair clinic twice a month.", "Omar",
        [ev("a1", "Are you still doing the repair clinic?"), ev("a2", "Yes, twice a month")], assertion="confirmation")])
    assert result.memories[0].support == SUPPORT_SOURCE
    assert result.memories[0].confirmed


@pytest.mark.parametrize(("statement", "quote", "code"), [
    ("Nadia owns a Tesla.", "Nadia", "UNSUPPORTED_CLAIM"),
    ("Nadia was paid 500 dollars for written work.", "I prefer written notes", "UNSUPPORTED_CLAIM"),
    ("Swimming relaxes everyone at sunrise.", "I prefer written notes", "QUOTE_UNRELATED"),
    ("Nadia loves swimming lakes sunrise.", "Nadia", "UNSUPPORTED_CLAIM"),
])
def test_unsupported_claims_are_rejected(statement, quote, code):
    result = run(window(NADIA), memories=[memory(statement, "Nadia", [ev("m1", quote)])])
    assert result.memories == []
    assert code in codes(result)


@pytest.mark.parametrize(("statement", "quote"), [
    ("Nadia does not prefer written notes.", "I prefer written notes"),
    ("Nadia is building Bluebird.", "I'm not building Bluebird"),
])
def test_changed_negation_needs_review(statement, quote):
    a = msg("m1", "user", "Nadia: I prefer written notes. I'm not building Bluebird.")
    result = run(window(a), memories=[memory(statement, "Nadia", [ev("m1", quote)])])
    assert "NEGATION_MISMATCH" in result.memories[0].review_reasons


def test_dropped_hedge_needs_review_but_a_kept_hedge_does_not():
    a = msg("a1", "assistant", "Omar: I might move the shop.")
    result = run(window(a), memories=[
        memory("Omar moves the shop.", "Omar", [ev("a1", "I might move the shop")]),
        memory("Omar might move the shop.", "Omar", [ev("a1", "I might move the shop")], "condition", "hypothetical")])
    assert result.memories[0].review_reasons == ("HEDGE_DROPPED",)
    assert result.memories[1].review_reasons == ()


@pytest.mark.parametrize(("evidence", "code"), [
    ([], "MISSING_EVIDENCE"),
    (None, "MISSING_EVIDENCE"),
    ([ev("ghost", "I prefer written notes")], "INVALID_EVIDENCE_MESSAGE"),
    ([ev("m1", "I prefer handwritten notes")], "QUOTE_NOT_IN_SOURCE"),
    ([ev("m1", "")], "QUOTE_NOT_IN_SOURCE"),
    (["m1"], "INVALID_EVIDENCE_MESSAGE"),
])
def test_missing_or_invalid_evidence_rejects_the_memory(evidence, code):
    result = run(window(NADIA), memories=[memory("Nadia prefers written notes.", "Nadia", evidence)])
    assert result.memories == []
    assert codes(result) == [code]


def test_context_message_alone_is_not_evidence():
    earlier = msg("a1", "user", "Nadia: I prefer written notes.", 0)
    target = msg("a2", "assistant", "Omar: Thanks.", 1)
    result = run(window(target, context=[earlier]), memories=[memory("Nadia prefers written notes.", "Nadia", [ev("a1", "I prefer written notes")])])
    assert result.memories == []
    assert codes(result) == ["CONTEXT_ONLY_EVIDENCE"]


def test_quote_matches_despite_case_spacing_and_curly_apostrophes():
    result = run(window(NADIA), memories=[memory("Nadia is building Bluebird.", "Nadia", [ev("m1", "I’M   building bluebird")])])
    assert len(result.memories) == 1


def test_entity_absent_from_its_message_is_rejected():
    result = run(window(NADIA), entities=[entity("Sparrow", "project", ["m1"]), entity("Nadia", "person", ["ghost"])])
    assert result.entities == []
    assert codes(result) == ["ENTITY_NOT_IN_SOURCE", "INVALID_EVIDENCE_MESSAGE"]


@pytest.mark.parametrize("raw", ["not json", "", None, "[]", '{"entities": [], "memories": []}',
                                 '{"entities": {}, "memories": [], "relationships": []}'])
def test_malformed_responses_fail_the_window(raw):
    result = validate_structured(raw, window(NADIA))
    assert result.status == "error"
    assert result.error
    assert result.raw == raw
    assert not (result.entities or result.memories or result.relationships)


def test_one_malformed_item_does_not_lose_the_rest():
    raw = json.dumps({"entities": ["bad"], "relationships": [7],
                      "memories": ["bad", memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")])]})
    result = validate_structured(raw, window(NADIA))
    assert len(result.memories) == 1
    assert codes(result).count("INVALID_SCHEMA") == 3
    assert result.status == "invalid"


def test_invalid_labels_are_marked_for_review_not_guessed():
    result = run(window(NADIA), memories=[memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")],
                                                  category="vibe", assertion="rumour", kind="someday")])
    m = result.memories[0]
    assert (m.category, m.assertion_type, m.temporal.kind) == ("unclassified", "unclassified", "unspecified")
    assert {"INVALID_CATEGORY", "INVALID_ASSERTION_TYPE"} <= set(m.review_reasons)


def test_truncated_response_fails_with_an_explicit_error():
    good = json.dumps({"entities": [], "memories": [], "relationships": []})
    result = validate_structured(good, window(NADIA), done_reason="length")
    assert result.status == "error"
    assert result.issues[0].code == "TRUNCATED_RESPONSE"
    assert result.raw == good


def test_duplicates_collapse_within_a_window():
    m = memory("Nadia prefers written notes.", "Nadia", [ev("m1", "I prefer written notes")])
    r = relationship("Nadia", "develops", "Bluebird", [ev("m1", "I'm building Bluebird")])
    e = entity("Nadia", "person", ["m1"])
    result = run(window(NADIA), [e, e], [m, m], [r, r])
    assert (len(result.entities), len(result.memories), len(result.relationships)) == (1, 1, 1)
    assert {"DUPLICATE_MEMORY", "DUPLICATE_RELATIONSHIP"} <= set(codes(result))


def test_examples_pass_the_validator_they_teach():
    prompt, output = STRUCTURED_EXAMPLES[0]
    assert "TARGET MESSAGES" in prompt
    w = window(
        Message("x2", CID, 1, "assistant", "Rafi: Yes, but I might leave in June. I chose Tallow over Ember because it is cheaper.", "2024-03-05"),
        Message("x3", CID, 2, "user", "Ines: Nice! I don't drink coffee anymore.", "2024-03-05"),
        context=[Message("x1", CID, 0, "user", "Ines: Rafi, are you still at Harbour Print?", "2024-03-05")])
    result = validate_structured(output, w)
    assert result.status == "ok"
    assert (len(result.entities), len(result.memories), len(result.relationships)) == (4, 4, 1)
    assert result.memories[0].speaker == "Rafi"


def test_prompt_and_examples_share_no_evaluation_names():
    text = STRUCTURED_SYSTEM_PROMPT + json.dumps(STRUCTURED_EXAMPLES)
    for name in ("Maya", "Leo", "Trailmark", "Caroline", "Melanie", "Pixel"):
        assert name not in text


class ScriptedClient:
    model_name = "scripted"

    def __init__(self, responses=None, fail=()):
        self.responses, self.fail = responses or {}, set(fail)
        self.calls, self.prompts = [], []
        self.last_usage = None

    def complete_json(self, system, prompt, schema, examples=()):
        ids = schema["properties"]["memories"]["items"]["properties"]["evidence"]["items"]["properties"]["message_id"]["enum"]
        self.calls.append(tuple(ids))
        self.prompts.append(prompt)
        index = len(self.calls) - 1
        if index in self.fail:
            raise AtomizationRequestError(f"scripted failure {index}")
        self.last_usage = {"prompt_eval_count": 100, "eval_count": 10, "done_reason": "stop"}
        return self.responses.get(index, json.dumps({"entities": [], "memories": [], "relationships": []}))


def conversation(n=9):
    return Conversation(CID, tuple(Message(f"m{i}", CID, i, "user", f"Nadia: Fact number {i} about Bluebird.", TS) for i in range(n)))


def good_response(mid="m0", text="Fact number 0 about Bluebird", statement="Nadia states fact number 0 about Bluebird."):
    return json.dumps({"entities": [entity("Bluebird", "project", [mid])],
                       "memories": [memory(statement, "Nadia", [ev(mid, text)])],
                       "relationships": []})


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "scratch.db"))
    yield s
    s.close()


def test_one_call_per_window_in_order(store):
    client = ScriptedClient()
    conv = conversation(9)
    report = ingest_structured(conv, store, StructuredExtractor(client))
    assert report.windows == report.model_calls == len(client.calls) == len(build_windows(conv)) == 3


def test_records_and_raw_output_are_stored(store):
    client = ScriptedClient({0: good_response()})
    report = ingest_structured(conversation(4), store, StructuredExtractor(client))
    sx = StructuredStore(store.connection)
    assert (report.entities, report.memories, report.relationships) == (1, 1, 0)
    assert sx.get_records(CID, "memory")[0]["memory_id"] == "c1:w0000:m1"
    assert sx.get_records(CID, "entity")[0]["name"] == "Bluebird"
    row = sx.get_windows(CID)[0]
    assert row["raw_output"] == good_response()
    assert row["usage"]["done_reason"] == "stop"


def test_failed_window_keeps_its_error_raw_output_and_earlier_results(store):
    client = ScriptedClient({0: good_response(), 1: "{broken"}, fail=[2])
    report = ingest_structured(conversation(9), store, StructuredExtractor(client))
    rows = StructuredStore(store.connection).get_windows(CID)
    assert report.failed_windows == 2
    assert report.memories == 1
    assert rows[1]["raw_output"] == "{broken"
    assert "invalid JSON" in rows[1]["error"]
    assert "scripted failure" in rows[2]["error"]


def test_prior_memories_give_identity_only_and_are_passed_forward(store):
    client = ScriptedClient({0: good_response()})
    ingest_structured(conversation(5), store, StructuredExtractor(client))
    assert "Nadia states fact number 0" in client.prompts[1]


def test_rerun_needs_replace_and_ids_stay_stable(store):
    conv = conversation(4)
    ingest_structured(conv, store, StructuredExtractor(ScriptedClient({0: good_response()})))
    first = StructuredStore(store.connection).get_records(CID, "memory")
    with pytest.raises(ValueError, match="replace=True"):
        ingest_structured(conv, store, StructuredExtractor(ScriptedClient()))
    ingest_structured(conv, store, StructuredExtractor(ScriptedClient({0: good_response()})), replace=True)
    assert StructuredStore(store.connection).get_records(CID, "memory") == first


def test_existing_conversation_chunks_are_not_replaced(store):
    conv = conversation(4)
    store.save_conversation(conv, [], __import__("numpy").zeros((0, 0), dtype="float32"), "none")
    store.connection.execute("UPDATE conversations SET title = 'kept' WHERE conversation_id = ?", (CID,))
    store.connection.commit()
    ingest_structured(conv, store, StructuredExtractor(ScriptedClient()))
    assert store.get_conversation(CID).title == "kept"
    changed = Conversation(CID, (Message("m0", CID, 0, "user", "different"),))
    with pytest.raises(ValueError, match="different messages"):
        ingest_structured(changed, store, StructuredExtractor(ScriptedClient()), replace=True)


def test_v2_tables_stay_empty_so_consolidation_and_retrieval_see_nothing(store):
    ingest_structured(conversation(4), store, StructuredExtractor(ScriptedClient({0: good_response()})))
    semantic = SemanticStore(store.connection)
    assert semantic.get_claims(CID) == []
    assert semantic.get_log(CID) == []


def test_existing_extract_modes_are_unchanged():
    assert EXTRACT_MODES == ("messages", "v2", "memories")


def test_factory_uses_the_existing_client_and_settings():
    extractor = structured_extractor(V2Settings(atomizer_model="some-model:1b", ollama_host="http://localhost:1"))
    assert extractor.model_name == "some-model:1b"
    assert extractor.client.host == "http://localhost:1"
