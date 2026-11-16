"""Structured extraction v1.1: target accountability, structural evidence checks and the review policy.

The model is scripted. These tests cover what code guarantees: where records are filed, what provenance
is verified, and that a model's reading is carried through without being re-judged on its wording. They
cannot show that a real model reads a conversation well; that needs a real-model evaluation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import continuator_memory.v2.structured_targets as targets_module
from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import AtomizationRequestError, V2Settings
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.structured_extraction import STRUCTURED_PROTOCOL_VERSION
from continuator_memory.v2.structured_store import StructuredStore
from continuator_memory.v2.structured_targets import (
    DEFAULT_TARGETS_NUM_PREDICT,
    OUTCOME_NOT_RETURNED,
    OUTCOME_UNPROCESSED,
    SEMANTIC_NOT_VERIFIED,
    STATUS_CONFLICTING,
    STATUS_EVIDENCE_VERIFIED,
    STATUS_NEEDS_CROSS_WINDOW,
    STATUS_UNCERTAIN,
    TARGETS_EXAMPLES,
    TARGETS_PROTOCOL_VERSION,
    TARGETS_SYSTEM_PROMPT,
    TargetExtractor,
    build_targets_prompt,
    ingest_targets,
    MEMORY_FIELDS,
    RELATIONSHIP_FIELDS,
    labels,
    normalize_response,
    target_extractor,
    targets_output_schema,
    unowned_messages,
    validate_targets,
)
from continuator_memory.v2.windowing import ContextWindow, build_windows

CID = "c1"
TS = "2024-05-10"


def msg(mid, role, text, position=0, ts=TS):
    return Message(mid, CID, position, role, text, ts)


def window(*targets, context=()):
    return ContextWindow(CID, 0, tuple(context), tuple(targets))


def mem(statement, quote, subject="Someone", predicate="does", obj="something", category="fact", stype="assertion",
        kind="current", subject_status="stated", object_status="stated", **extra):
    return {"statement": statement, "subject": subject, "subject_status": subject_status, "predicate": predicate,
            "object": obj, "object_status": object_status, "category": category, "statement_type": stype,
            "time_kind": kind, "quote": quote, **extra}


def link(subject, rtype, obj, quote, stype="assertion", kind="current", **extra):
    return {"subject": subject, "type": rtype, "object": obj, "statement_type": stype, "time_kind": kind, "quote": quote, **extra}


def slot(*memories, relationships=(), outcome="extracted", **extra):
    return {"outcome": outcome, "memories": list(memories), "relationships": list(relationships), **extra}


def run(w, targets, entities=(), covered=(), prior=()):
    return validate_targets(json.dumps({"entities": list(entities), "targets": targets}), w, prior, covered=covered)


def codes(result):
    return [i.code for i in result.issues]


# ---- 1, 2. targets, context and coverage ------------------------------------

def test_targets_and_context_get_separate_local_labels():
    w = window(msg("a", "user", "Ana: one"), msg("b", "user", "Ana: two"), context=[msg("z", "assistant", "Bram: zero")])
    assert {k: m.message_id for k, m in labels(w).items()} == {"T1": "a", "T2": "b", "C1": "z"}
    prompt = build_targets_prompt(w)
    assert "[T1]" in prompt and "[C1]" in prompt
    for global_id in ("a:", "z:"):
        assert f"[{global_id}" not in prompt


def test_schema_requires_exactly_one_slot_per_target_and_none_for_context():
    w = window(msg("a", "user", "Ana: one"), msg("b", "user", "Ana: two"), context=[msg("z", "user", "Ana: zero")])
    slots = targets_output_schema(w)["properties"]["targets"]
    assert slots["required"] == ["T1", "T2"]
    assert set(slots["properties"]) == {"T1", "T2"}
    assert slots["additionalProperties"] is False


def test_schema_has_no_global_ids_or_global_state():
    w = window(msg("secret-global-id", "user", "Ana: one"))
    text = json.dumps(targets_output_schema(w))
    for word in ("secret-global-id", "superseded", "is_current", "memory_id", "confirmed"):
        assert word not in text


@pytest.mark.parametrize(("n", "per_window", "context"), [(1, 4, 4), (9, 4, 4), (62, 4, 4), (7, 3, 1), (10, 1, 0), (5, 2, 9)])
def test_every_message_is_a_target_exactly_once(n, per_window, context):
    conv = Conversation(CID, tuple(msg(f"m{i}", "user", f"Ana: {i}", i) for i in range(n)))
    assert unowned_messages(conv, per_window, context) == []
    assert sum(len(w.targets) for w in build_windows(conv, per_window, context)) == n


def test_each_target_gets_a_recorded_outcome():
    w = window(msg("a", "user", "Ana: Thanks!"), msg("b", "user", "Ana: What time is it?"), msg("c", "user", "Ana: Hmm, the thing."))
    result = run(w, {"T1": slot(outcome="nothing_durable"), "T2": slot(outcome="question"), "T3": slot(outcome="unclear", note="no referent")})
    assert [(o.message_id, o.outcome) for o in result.outcomes] == [("a", "nothing_durable"), ("b", "question"), ("c", "unclear")]
    assert result.outcomes[2].note == "no referent"
    assert result.unaccounted_targets == ()


def test_a_skipped_target_is_visible_not_silent():
    w = window(msg("a", "user", "Ana: I bake."), msg("b", "user", "Ana: I knit."))
    result = run(w, {"T1": slot(mem("Ana bakes.", "I bake", "Ana"))})
    assert result.unaccounted_targets == ("b",)
    assert result.outcomes[1].outcome == OUTCOME_NOT_RETURNED
    assert "MISSING_TARGET_OUTCOME" in codes(result)
    assert len(result.memories) == 1


def test_record_for_a_context_label_is_reported():
    w = window(msg("a", "user", "Ana: I bake."), context=[msg("z", "user", "Ana: I knit.")])
    result = run(w, {"T1": slot(outcome="nothing_durable"), "C1": slot(mem("Ana knits.", "I knit", "Ana"))})
    assert result.memories == []
    assert "UNKNOWN_TARGET" in codes(result)


def test_filed_under_the_wrong_target_moves_to_its_true_owner():
    w = window(msg("a", "user", "Ana: Hello."), msg("b", "assistant", "Bram: I repair clocks."))
    result = run(w, {"T1": slot(mem("Bram repairs clocks.", "I repair clocks", "Bram")), "T2": slot(outcome="nothing_durable")})
    m = result.memories[0]
    assert (m.message_id, m.speaker, m.speaker_role) == ("b", "Bram", "assistant")
    assert "FILED_UNDER_WRONG_TARGET" in codes(result)


# ---- 3. pronouns ------------------------------------------------------------

def test_context_resolves_a_pronoun_but_the_source_stays_the_target():
    earlier = msg("z", "assistant", "Bram: How is the Kestrel engine?", 0)
    target = msg("a", "user", "Ana: It failed its bench test.", 1)
    result = run(window(target, context=[earlier]), {"T1": slot(mem(
        "The Kestrel engine failed its bench test.", "It failed its bench test", "the Kestrel engine",
        "failed", "its bench test", subject_status="resolved_from_context", kind="historical",
        context=[{"ref": "C1", "quote": "How is the Kestrel engine?"}]))})
    m = result.memories[0]
    assert m.message_id == "a"
    assert m.message_ids == ("a", "z")
    assert m.subject_status == "resolved_from_context"
    assert m.status == STATUS_EVIDENCE_VERIFIED


def test_reference_the_model_cannot_resolve_stays_explicitly_unresolved():
    target = msg("a", "user", "Ana: They cancelled it.")
    result = run(window(target), {"T1": slot(mem("Someone cancelled something.", "They cancelled it", "", "cancelled", "",
                                                  subject_status="unresolved", object_status="unresolved", kind="historical"))})
    m = result.memories[0]
    assert (m.subject, m.object, m.status) == (None, None, STATUS_NEEDS_CROSS_WINDOW)
    assert {"SUBJECT_UNRESOLVED", "OBJECT_UNRESOLVED"} <= set(m.review_reasons)


# ---- 4. overlapping windows: the m31 failure mode ---------------------------

def test_fact_skipped_in_its_own_window_is_recovered_from_context_not_lost():
    earlier = msg("z", "assistant", "Bram: The Oak Street deal collapsed.", 0)
    target = msg("a", "user", "Ana: Oh no.", 1)
    result = run(window(target, context=[earlier]), {"T1": slot(
        mem("The Oak Street deal collapsed.", "The Oak Street deal collapsed", "the Oak Street deal", "collapsed", "",
            object_status="not_applicable", kind="historical"), outcome="nothing_durable")})
    m = result.memories[0]
    assert (m.message_id, m.speaker, m.status) == ("z", "Bram", STATUS_NEEDS_CROSS_WINDOW)
    assert m.review_reasons == ("RECOVERED_FROM_CONTEXT",)


def test_repeat_of_an_already_extracted_context_message_is_dropped():
    earlier = msg("z", "assistant", "Bram: The Oak Street deal collapsed.", 0)
    target = msg("a", "user", "Ana: Oh no.", 1)
    result = run(window(target, context=[earlier]), {"T1": slot(
        mem("The Oak Street deal collapsed.", "The Oak Street deal collapsed", "the Oak Street deal", kind="historical"))},
        covered={"z"})
    assert result.memories == []
    assert codes(result) == ["CONTEXT_REPEAT_IGNORED"]
    assert result.status == "ok"


class ScriptedClient:
    model_name = "scripted"

    def __init__(self, responses=None, fail=(), usage=None):
        self.responses, self.fail, self.usage = responses or {}, set(fail), usage or {}
        self.calls = 0
        self.last_usage = None

    def complete_json(self, system, prompt, schema, examples=()):
        index = self.calls
        self.calls += 1
        if index in self.fail:
            raise AtomizationRequestError(f"scripted failure {index}")
        self.last_usage = self.usage.get(index, {"prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"})
        slots = schema["properties"]["targets"]["required"]
        default = json.dumps({"entities": [], "targets": {k: {"outcome": "nothing_durable"} for k in slots}})
        return self.responses.get(index, default)


def conversation(*texts):
    return Conversation(CID, tuple(msg(f"m{i}", "user", t, i) for i, t in enumerate(texts)))


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "scratch.db"))
    yield s
    s.close()


def test_runner_recovers_a_skipped_fact_and_drops_later_repeats(store):
    conv = conversation("Ana: The pier reopened.", "Ana: Nice day.", "Ana: Yes.", "Ana: Bye.", "Ana: Hi.", "Ana: Hey.")
    recovered = json.dumps({"entities": [], "targets": {
        "T1": slot(mem("The pier reopened.", "The pier reopened", "the pier", "reopened", "", object_status="not_applicable", kind="historical"),
                   outcome="nothing_durable"), "T2": slot(outcome="nothing_durable")}})
    extractor = TargetExtractor(ScriptedClient({1: recovered}))
    report = ingest_targets(conv, store, extractor, targets_per_window=2, context_before=2)
    records = StructuredStore(store.connection).get_records(CID, "memory")
    assert report.memories == 1
    assert records[0]["message_id"] == "m0"
    assert records[0]["status"] == STATUS_NEEDS_CROSS_WINDOW


# ---- 5. questions and confirmations -----------------------------------------

def test_question_is_an_outcome_and_the_schema_cannot_call_a_memory_a_question():
    w = window(msg("a", "assistant", "Bram: Do you still sail at Port Lune?"))
    schema = targets_output_schema(w)
    slot_schema = schema["properties"]["targets"]["properties"]["T1"]["properties"]
    assert "question" in slot_schema["outcome"]["enum"]
    assert "question" not in slot_schema["memories"]["items"]["properties"]["statement_type"]["enum"]
    result = run(w, {"T1": slot(outcome="question")})
    assert result.memories == []
    assert result.outcomes[0].outcome == "question"


def test_question_and_reply_jointly_support_a_confirmed_relationship():
    question = msg("z", "assistant", "Bram: Are you still with the Lune Sailing Club?", 0)
    reply = msg("a", "user", "Ana: Yes.", 1)
    cite = [{"ref": "C1", "quote": "Are you still with the Lune Sailing Club?"}]
    result = run(window(reply, context=[question]), {"T1": slot(
        mem("Ana is with the Lune Sailing Club.", "Yes", "Ana", "is with", "the Lune Sailing Club", stype="confirmation", context=cite),
        relationships=[link("Ana", "member_of", "the Lune Sailing Club", "Yes", "confirmation", context=cite)])})
    assert result.memories[0].speaker == "Ana"
    assert result.memories[0].statement_type == "confirmation"
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED
    assert result.relationships[0].message_ids == ("a", "z")


def test_confirmation_with_nothing_to_confirm_is_uncertain():
    a = msg("a", "user", "Ana: Yes, I do.")
    result = run(window(a), {"T1": slot(mem("Ana sails.", "Yes, I do", "Ana", stype="confirmation"))})
    assert result.memories[0].status == STATUS_UNCERTAIN
    assert result.memories[0].review_reasons == ("CONFIRMATION_WITHOUT_PROMPT",)


def test_records_on_a_question_outcome_are_kept_and_flagged():
    a = msg("a", "assistant", "Bram: Do you sail?")
    result = run(window(a), {"T1": slot(mem("Bram asks about sailing.", "Do you sail?", "Bram"), outcome="question")})
    assert "OUTCOME_CONFLICT" in codes(result)
    assert len(result.memories) == 1


# ---- 6. negation and confirmation ------------------------------------------

@pytest.mark.parametrize(("source", "statement", "quote"), [
    ("Zuri: Yes. That hasn't changed.", "The ferry still leaves at noon.", "That hasn't changed"),
    ("Zuri: I never eat peanuts.", "Zuri does not eat peanuts.", "I never eat peanuts"),
    ("Zuri: Nothing is wrong with the kiln.", "The kiln works fine.", "Nothing is wrong with the kiln"),
])
def test_negation_wording_is_not_judged_by_code(source, statement, quote):
    a = msg("a", "user", source)
    result = run(window(a), {"T1": slot(mem(statement, quote, "Zuri"))})
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED
    assert result.memories[0].review_reasons == ()


def test_a_flipped_meaning_is_not_detected_and_says_so():
    """Documents the limit: code proves the quote, not the reading, so the record says it is not verified."""
    a = msg("a", "user", "Zuri: I am not moving to Oslo.")
    result = run(window(a), {"T1": slot(mem("Zuri is moving to Oslo.", "I am not moving to Oslo", "Zuri", kind="future"))})
    data = result.memories[0].to_dict()
    assert data["status"] == STATUS_EVIDENCE_VERIFIED
    assert data["semantic_support"] == SEMANTIC_NOT_VERIFIED
    assert "confirmed" not in data


# ---- 7. hypothetical versus actual ------------------------------------------

def test_hypothetical_and_actual_stay_distinct_on_one_message():
    a = msg("a", "assistant", "Bram: I sold the van. I could buy a truck if the bank agrees.")
    result = run(window(a), {"T1": slot(
        mem("Bram sold the van.", "I sold the van", "Bram", "sold", "the van", "event", kind="historical"),
        mem("Bram could buy a truck if the bank agrees.", "I could buy a truck if the bank agrees", "Bram", "could buy", "a truck",
            "condition", "hypothetical", "conditional"))})
    assert [(m.statement_type, m.temporal.kind) for m in result.memories] == [("assertion", "historical"), ("hypothetical", "conditional")]
    assert all(m.status == STATUS_EVIDENCE_VERIFIED for m in result.memories)


def test_suggestion_and_report_keep_their_type():
    a = msg("a", "assistant", "Bram: You should try cedar. Priti told me she left Oslo.")
    result = run(window(a), {"T1": slot(
        mem("Bram suggests trying cedar.", "You should try cedar", "Bram", "suggests trying", "cedar", stype="suggestion"),
        mem("Priti left Oslo, according to Bram.", "Priti told me she left Oslo", "Priti", "left", "Oslo", stype="reported", kind="historical"))})
    assert [m.statement_type for m in result.memories] == ["suggestion", "reported"]


# ---- 8. speakers ------------------------------------------------------------

def test_speaker_comes_from_the_message_whatever_the_model_writes():
    a = msg("a", "assistant", "Bram: Priti runs the bakery.")
    result = run(window(a), {"T1": slot(mem("Priti runs the bakery.", "Priti runs the bakery", "Priti", stype="reported"))})
    assert (result.memories[0].speaker, result.memories[0].speaker_role) == ("Bram", "assistant")


def test_model_cannot_supply_a_speaker():
    a = msg("a", "assistant", "Bram: Priti runs the bakery.")
    item = mem("Priti runs the bakery.", "Priti runs the bakery", "Priti", stype="reported")
    item["speaker"] = "Priti"
    result = run(window(a), {"T1": slot(item)})
    assert result.memories == []
    assert "UNEXPECTED_KEY" in codes(result)


def test_unlabelled_message_uses_its_role():
    a = msg("a", "user", "I keep bees.")
    result = run(window(a), {"T1": slot(mem("The user keeps bees.", "I keep bees", "the user"))})
    assert result.memories[0].speaker == "user"
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED


# ---- 9. semantic variation --------------------------------------------------

@pytest.mark.parametrize(("source", "statement", "quote"), [
    ("Mei: Please send me printed agendas.", "Mei prefers receiving printed agendas.", "send me printed agendas"),
    ("Mei: Pottery is my favourite thing.", "Mei enjoys ceramics most of all.", "Pottery is my favourite thing"),
    ("Mei: I've got a strong liking for rye.", "Mei prefers rye.", "strong liking for rye"),
    ("Mei: I fly out once the visa clears.", "Mei will travel after receiving her visa approval.", "I fly out once the visa clears"),
])
def test_different_wording_alone_never_triggers_review(source, statement, quote):
    a = msg("a", "user", source)
    result = run(window(a), {"T1": slot(mem(statement, quote, "Mei"))})
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED
    assert not [c for c in codes(result) if "LEXICAL" in c]


def test_model_uncertainty_is_preserved_with_its_reason():
    a = msg("a", "user", "Mei: Sure, that could work, I guess.")
    result = run(window(a), {"T1": slot(mem("Mei accepts the plan.", "that could work", "Mei", uncertain="Reluctant; may not be agreement."))})
    m = result.memories[0]
    assert m.status == STATUS_UNCERTAIN
    assert m.model_uncertainty == "Reluctant; may not be agreement."


# ---- 10. missing subjects, objects and predicates ---------------------------

def test_event_subject_and_no_object_are_representable():
    a = msg("a", "user", "Teo: The harvest festival moved to October.")
    result = run(window(a), {"T1": slot(mem("The harvest festival moved to October.", "The harvest festival moved to October",
                                             "the harvest festival", "moved to", "October", "event", kind="historical"),
                                         mem("Teo left.", "Teo", "Teo", "left", "", object_status="not_applicable", kind="historical"))})
    assert result.memories[0].subject == "the harvest festival"
    assert (result.memories[1].object, result.memories[1].object_status, result.memories[1].status) == (None, "not_applicable", STATUS_EVIDENCE_VERIFIED)


@pytest.mark.parametrize(("field", "reason"), [("subject", "SUBJECT_MISSING"), ("object", "OBJECT_MISSING")])
def test_empty_required_field_is_not_filled_in(field, reason):
    a = msg("a", "user", "Teo: There are optional trips to the depot.")
    item = mem("There are optional trips to the depot.", "There are optional trips to the depot", "optional trips", "go to", "the depot")
    item[field] = ""
    result = run(window(a), {"T1": slot(item)})
    m = result.memories[0]
    assert getattr(m, field) is None
    assert m.status == STATUS_UNCERTAIN
    assert reason in m.review_reasons


@pytest.mark.parametrize("predicate", ["", "'s", "'re going"])
def test_fragment_predicate_is_kept_and_marked(predicate):
    a = msg("a", "user", "Teo: My kayak is red.")
    result = run(window(a), {"T1": slot(mem("Teo's kayak is red.", "My kayak is red", "Teo", predicate, "red"))})
    assert result.memories[0].status == STATUS_UNCERTAIN
    assert "INCOMPLETE_PREDICATE" in result.memories[0].review_reasons


def test_subject_named_nowhere_needs_cross_window_context_and_is_not_rejected():
    a = msg("a", "user", "Teo: It closes at nine.")
    result = run(window(a), {"T1": slot(mem("Marlowe's closes at nine.", "It closes at nine", "Marlowe's", subject_status="resolved_from_context"))})
    assert result.memories[0].status == STATUS_NEEDS_CROSS_WINDOW
    assert result.memories[0].review_reasons == ("REFERENCE_NOT_IN_WINDOW",)


# ---- 11. temporal -----------------------------------------------------------

def test_each_record_has_its_own_kind_on_one_message():
    a = msg("a", "user", "Ivo: I used to row, I swim now, and I will cycle in May.")
    result = run(window(a), {"T1": slot(
        mem("Ivo used to row.", "I used to row", "Ivo", kind="historical"),
        mem("Ivo swims.", "I swim now", "Ivo", kind="current"),
        mem("Ivo will cycle in May.", "I will cycle in May", "Ivo", "will cycle", kind="future",
            time_expression="in May", time_value="2024-05"))})
    assert [m.temporal.kind for m in result.memories] == ["historical", "current", "future"]


def test_ambiguous_date_stays_unnormalized_and_uncertain():
    a = msg("a", "user", "Ivo: We moved the launch to next spring, maybe.")
    result = run(window(a), {"T1": slot(mem("The launch may be next spring.", "We moved the launch to next spring, maybe", "the launch",
                                             kind="unspecified", time_expression="next spring", uncertain="Year and date are unclear."))})
    t = result.memories[0].temporal
    assert (t.expression, t.normalized, t.kind) == ("next spring", None, "unspecified")
    assert result.memories[0].status == STATUS_UNCERTAIN


def test_message_date_is_not_used_as_the_fact_date():
    a = msg("a", "user", "Ivo: I started at the mill.", ts="2024-05-10")
    result = run(window(a), {"T1": slot(mem("Ivo started at the mill.", "I started at the mill", "Ivo", kind="historical"))})
    assert result.memories[0].temporal.normalized is None
    assert result.memories[0].temporal.expression is None


@pytest.mark.parametrize(("kind", "value", "ts", "status", "code"), [
    ("historical", "2025-01", "2024-05-10", STATUS_CONFLICTING, "TEMPORAL_CONFLICT"),
    ("future", "2023-01", "2024-05-10", STATUS_CONFLICTING, "TEMPORAL_CONFLICT"),
    ("future", "2024-13-40", "2024-05-10", STATUS_UNCERTAIN, "MALFORMED_TEMPORAL"),
    ("future", "2024-06 to 2024-08", "2024-05-10", STATUS_UNCERTAIN, "MALFORMED_TEMPORAL"),
    ("future", "2024-06", None, STATUS_UNCERTAIN, "RESOLVED_WITHOUT_TIMESTAMP"),
])
def test_impossible_or_unsupported_dates_are_surfaced(kind, value, ts, status, code):
    a = msg("a", "user", "Ivo: The dam opens in June.", ts=ts)
    result = run(window(a), {"T1": slot(mem("The dam opens in June.", "The dam opens in June", "the dam", kind=kind,
                                             time_expression="in June", time_value=value))})
    assert result.memories[0].status == status
    assert code in codes(result)


def test_expression_not_in_cited_text_is_dropped_not_invented():
    a = msg("a", "user", "Ivo: The dam opened.")
    result = run(window(a), {"T1": slot(mem("The dam opened last week.", "The dam opened", "the dam", kind="historical",
                                             time_expression="last week", time_value="2024-05"))})
    t = result.memories[0].temporal
    assert (t.expression, t.normalized) == (None, None)
    assert result.memories[0].status == STATUS_UNCERTAIN


# ---- 12. corrections --------------------------------------------------------

def test_correction_is_distinct_and_supersedes_nothing():
    earlier = msg("z", "user", "Lumi: The clinic opens Monday.", 0)
    fix = msg("a", "user", "Lumi: Sorry, I meant Tuesday.", 1)
    result = run(window(fix, context=[earlier]), {"T1": slot(mem(
        "The clinic opens Tuesday, not Monday.", "I meant Tuesday", "the clinic", "opens", "Tuesday", stype="correction", kind="correction",
        subject_status="resolved_from_context", context=[{"ref": "C1", "quote": "The clinic opens Monday"}]))})
    data = result.memories[0].to_dict()
    assert (data["statement_type"], data["temporal"]["kind"]) == ("correction", "correction")
    assert not {"superseded", "superseded_by", "is_current", "latest"} & set(data)


# ---- 13. reasons ------------------------------------------------------------

def test_decision_keeps_its_reason_with_evidence():
    a = msg("a", "user", "Noor: I dropped the grant because the reporting took too long.")
    result = run(window(a), {"T1": slot(mem("Noor dropped the grant.", "I dropped the grant", "Noor", "dropped", "the grant", "decision", kind="historical",
                                             reason="The reporting took too long.", reason_quote="because the reporting took too long"))})
    assert result.memories[0].reason == "The reporting took too long."
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED


def test_reason_from_a_cited_context_message_is_allowed():
    earlier = msg("z", "user", "Noor: The reporting took too long.", 0)
    a = msg("a", "user", "Noor: So I dropped the grant.", 1)
    result = run(window(a, context=[earlier]), {"T1": slot(mem(
        "Noor dropped the grant.", "I dropped the grant", "Noor", "dropped", "the grant", "decision", kind="historical",
        reason="The reporting took too long.", reason_quote="The reporting took too long",
        context=[{"ref": "C1", "quote": "The reporting took too long"}]))})
    assert result.memories[0].reason


def test_reason_without_evidence_is_marked_and_invented_quote_is_removed():
    a = msg("a", "user", "Noor: I dropped the grant.")
    no_quote = mem("Noor dropped the grant.", "I dropped the grant", "Noor", "dropped", "the grant", "decision", reason="Too slow.")
    bad_quote = mem("Noor gave up the grant.", "I dropped the grant", "Noor", "gave up", "the grant", "decision",
                    reason="Licensing fees.", reason_quote="licensing fees were high")
    result = run(window(a), {"T1": slot(no_quote, bad_quote)})
    assert result.memories[0].review_reasons == ("REASON_WITHOUT_EVIDENCE",)
    assert result.memories[1].reason is None
    assert result.memories[1].review_reasons == ("REASON_QUOTE_NOT_IN_SOURCE",)


# ---- 14. duplicates ---------------------------------------------------------

def test_duplicates_within_a_window_collapse_structurally():
    a = msg("a", "user", "Sol: I fence on Fridays.")
    m = mem("Sol fences on Fridays.", "I fence on Fridays", "Sol")
    r = link("Sol", "attends", "fencing", "I fence on Fridays")
    e = {"name": "Sol", "type": "person"}
    result = run(window(a), {"T1": slot(m, m, relationships=[r, r])}, entities=[e, e])
    assert (len(result.memories), len(result.relationships), len(result.entities)) == (1, 1, 1)
    assert {"DUPLICATE_MEMORY", "DUPLICATE_RELATIONSHIP", "DUPLICATE_ENTITY"} <= set(codes(result))


def test_entity_mentions_are_computed_by_code():
    a, b = msg("a", "user", "Sol: Rook Press called."), msg("b", "assistant", "Bram: Rook Press again?")
    result = run(window(a, b), {"T1": slot(outcome="nothing_durable"), "T2": slot(outcome="question")},
                 entities=[{"name": "Rook Press", "type": "company"}])
    assert result.entities[0].message_ids == ("a", "b")
    assert result.entities[0].speakers == ("Sol", "Bram")


def test_entity_known_from_context_is_kept_without_claiming_a_target_mention():
    context = msg("z", "user", "Sol: Rook Press called.", 0)
    target = msg("a", "user", "Sol: They want a reprint.", 1)
    result = run(window(target, context=[context]), {"T1": slot(outcome="nothing_durable")}, entities=[{"name": "Rook Press", "type": "company"}])
    assert result.entities[0].message_ids == ()


def test_entity_named_nowhere_and_unsupported_alias_are_refused():
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(outcome="nothing_durable")},
                 entities=[{"name": "Falcon Books", "type": "company"}, {"name": "Rook Press", "type": "company", "aliases": ["Rook & Co"]}])
    assert [e.name for e in result.entities] == ["Rook Press"]
    assert result.entities[0].aliases == ()
    assert {"ENTITY_NOT_IN_WINDOW", "UNSUPPORTED_ALIAS"} <= set(codes(result))


# ---- 15. unsupported relationships and evidence -----------------------------

@pytest.mark.parametrize(("item", "code"), [
    (link("Sol", "owns", "Rook Press", ""), "MISSING_EVIDENCE"),
    (link("Sol", "owns", "Rook Press", "I bought Rook Press yesterday"), "QUOTE_NOT_IN_SOURCE"),
    (link("Sol", "", "Rook Press", "Rook Press called"), "INVALID_RELATIONSHIP_TYPE"),
    (link("Sol", "owns things!", "Rook Press", "Rook Press called"), "INVALID_RELATIONSHIP_TYPE"),
    (link("Sol", "owns", "", "Rook Press called"), "EMPTY_RELATIONSHIP"),
    (link("Sol", "owns", "Rook Press", "Rook Press called", context=[{"ref": "C9", "quote": "x"}]), "INVALID_EVIDENCE_REF"),
    (link("Sol", "owns", "Rook Press", "Rook Press called", context=[{"ref": "T1", "quote": "not there"}]), "QUOTE_NOT_IN_SOURCE"),
])
def test_relationship_with_bad_evidence_is_rejected(item, code):
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(relationships=[item])})
    assert result.relationships == []
    assert code in codes(result)


def test_open_relationship_vocabulary():
    a = msg("a", "user", "Sol: Rook Press prints for Ivy Books.")
    result = run(window(a), {"T1": slot(relationships=[link("Rook Press", "Prints For", "Ivy Books", "Rook Press prints for Ivy Books")])})
    assert result.relationships[0].type == "prints_for"


def test_relationship_endpoint_named_nowhere_is_not_treated_as_established():
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(relationships=[link("Rook Press", "owned_by", "Falcon Group", "Rook Press called")])})
    assert result.relationships[0].status == STATUS_NEEDS_CROSS_WINDOW


def test_co_occurring_names_with_a_real_quote_are_not_detected_and_say_so():
    """Documents the limit: code cannot read meaning, so the record stays explicitly unverified."""
    a = msg("a", "user", "Sol: Rook Press called, and Ivy Books emailed.")
    result = run(window(a), {"T1": slot(relationships=[link("Rook Press", "partners_with", "Ivy Books", "Rook Press called, and Ivy Books emailed")])})
    assert result.relationships[0].to_dict()["semantic_support"] == SEMANTIC_NOT_VERIFIED


@pytest.mark.parametrize(("quote", "code"), [("", "MISSING_EVIDENCE"), ("Sol invented this quote", "QUOTE_NOT_IN_SOURCE")])
def test_memory_with_bad_evidence_is_rejected_and_the_claim_stays_visible(quote, code):
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(mem("Sol bought a horse.", quote, "Sol"))})
    assert result.memories == []
    assert code in codes(result)
    assert "Sol bought a horse." in result.issues[0].detail


def test_empty_statement_is_structurally_invalid():
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(mem("", "Rook Press called", "Sol"))})
    assert codes(result) == ["EMPTY_STATEMENT"]


def test_invalid_labels_are_marked_not_guessed():
    a = msg("a", "user", "Sol: Rook Press called.")
    result = run(window(a), {"T1": slot(mem("Rook Press called Sol.", "Rook Press called", "Rook Press", "called", "Sol", "vibe", "rumour", "someday",
                                             subject_status="???", object_status="???"))})
    m = result.memories[0]
    assert (m.category, m.statement_type, m.temporal.kind) == ("unclassified", "unclassified", "unspecified")
    assert m.needs_review


# ---- 16. truncation and malformed responses ---------------------------------

def test_truncated_output_fails_explicitly_with_targets_unprocessed():
    w = window(msg("a", "user", "Sol: One."), msg("b", "user", "Sol: Two."))
    good = json.dumps({"entities": [], "targets": {"T1": slot(outcome="nothing_durable"), "T2": slot(outcome="nothing_durable")}})
    result = validate_targets(good, w, done_reason="length")
    assert result.status == "error"
    assert result.issues[0].code == "TRUNCATED_RESPONSE"
    assert result.raw == good
    assert [o.outcome for o in result.outcomes] == [OUTCOME_UNPROCESSED, OUTCOME_UNPROCESSED]
    assert result.unaccounted_targets == ("a", "b")


@pytest.mark.parametrize("raw", ["not json", "", None, "[]", '{"entities": []}', '{"entities": {}, "targets": {}}', '{"entities": [], "targets": []}'])
def test_malformed_responses_fail_the_window_and_keep_the_raw_text(raw):
    w = window(msg("a", "user", "Sol: One."))
    result = validate_targets(raw, w)
    assert result.status == "error"
    assert result.raw == raw
    assert result.unaccounted_targets == ("a",)


def test_one_malformed_item_does_not_lose_the_rest():
    a = msg("a", "user", "Sol: I fence.")
    result = run(window(a), {"T1": {"outcome": "extracted", "memories": ["bad", mem("Sol fences.", "I fence", "Sol")], "relationships": [7]}}, entities=["bad"])
    assert len(result.memories) == 1
    assert codes(result).count("INVALID_SCHEMA") == 2
    assert "EMPTY_ENTITY" in codes(result)


def test_truncation_is_reported_and_later_windows_continue(store):
    conv = conversation("Ana: A.", "Ana: B.", "Ana: C.", "Ana: D.")
    client = ScriptedClient(usage={0: {"prompt_eval_count": 1, "eval_count": 4096, "done_reason": "length"}}, fail=[1])
    report = ingest_targets(conv, store, TargetExtractor(client), targets_per_window=2, context_before=0)
    assert client.calls == 2
    assert report.truncated_windows == 1
    assert report.failed_windows == 2
    assert report.unaccounted_targets == ["m0", "m1", "m2", "m3"]
    sx = StructuredStore(store.connection)
    assert {o["outcome"] for o in sx.get_outcomes(CID).values()} == {OUTCOME_UNPROCESSED}
    assert "scripted failure" in sx.get_windows(CID)[1]["error"]


# ---- pipeline, storage and isolation ----------------------------------------

def test_one_call_per_window_and_outcomes_for_every_message(store):
    conv = conversation(*[f"Ana: {i}." for i in range(9)])
    client = ScriptedClient()
    report = ingest_targets(conv, store, TargetExtractor(client))
    assert report.windows == report.model_calls == client.calls == 3
    assert set(StructuredStore(store.connection).get_outcomes(CID)) == {m.message_id for m in conv.messages}


def test_records_store_protocol_and_raw_output(store):
    conv = conversation("Ana: I bake.")
    raw = json.dumps({"entities": [{"name": "Ana", "type": "person"}], "targets": {"T1": slot(mem("Ana bakes.", "I bake", "Ana"))}})
    ingest_targets(conv, store, TargetExtractor(ScriptedClient({0: raw})))
    sx = StructuredStore(store.connection)
    row = sx.get_windows(CID)[0]
    assert (row["protocol"], row["raw_output"]) == (TARGETS_PROTOCOL_VERSION, raw)
    assert sx.get_records(CID, "memory")[0]["memory_id"] == "c1:w0000:m1"


def test_rerun_needs_replace(store):
    conv = conversation("Ana: I bake.")
    ingest_targets(conv, store, TargetExtractor(ScriptedClient()))
    with pytest.raises(ValueError, match="replace=True"):
        ingest_targets(conv, store, TargetExtractor(ScriptedClient()))
    ingest_targets(conv, store, TargetExtractor(ScriptedClient()), replace=True)


def test_v2_tables_stay_empty_so_retrieval_sees_nothing(store):
    raw = json.dumps({"entities": [], "targets": {"T1": slot(mem("Ana bakes.", "I bake", "Ana"))}})
    ingest_targets(conversation("Ana: I bake."), store, TargetExtractor(ScriptedClient({0: raw})))
    assert SemanticStore(store.connection).get_claims(CID) == []


def test_phase1_protocol_default_is_unchanged():
    assert STRUCTURED_PROTOCOL_VERSION == "structured-v1"
    assert TARGETS_PROTOCOL_VERSION != STRUCTURED_PROTOCOL_VERSION


def test_output_limit_default_and_env_override(monkeypatch):
    settings = V2Settings(atomizer_model="m", ollama_host="http://localhost:1", num_predict=2048)
    monkeypatch.delenv("CONTINUATOR_V2_NUM_PREDICT", raising=False)
    assert target_extractor(settings).client.num_predict == DEFAULT_TARGETS_NUM_PREDICT
    monkeypatch.setenv("CONTINUATOR_V2_NUM_PREDICT", "2048")
    assert target_extractor(settings).client.num_predict == 2048


# ---- the design rules themselves --------------------------------------------

def test_example_passes_the_validator_it_teaches():
    w = window(
        Message("t1", CID, 1, "assistant", "Rafi: Yes, but I might leave in June. I chose Tallow over Ember because it is cheaper.", "2024-03-05"),
        Message("t2", CID, 2, "user", "Ines: Nice! Actually it's Thursday I swim, not Tuesday. Do you still cycle?", "2024-03-05"),
        context=[Message("c1", CID, 0, "user", "Ines: Rafi, are you still at Harbour Print?", "2024-03-05")])
    result = validate_targets(TARGETS_EXAMPLES[0][1], w)
    assert result.status == "ok"
    assert [i.code for i in result.issues] == []
    assert all(m.status == STATUS_EVIDENCE_VERIFIED for m in result.memories)
    assert [m.temporal.kind for m in result.memories] == ["current", "conditional", "historical", "correction"]


def test_prompt_has_no_evaluation_entities():
    text = TARGETS_SYSTEM_PROMPT + json.dumps(TARGETS_EXAMPLES)
    for name in ("Maya", "Leo", "Trailmark", "FieldNotes", "Northstar", "GreenCurrent", "Hawthorne", "Caroline", "Melanie", "Pixel"):
        assert name not in text


def test_module_has_no_language_regexes_stemming_or_word_overlap():
    tree = ast.parse(Path(targets_module.__file__).read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.alias)} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "re" not in names
    for banned in ("tokens", "FRAME_TOKENS", "content_tokens", "classify_support", "NAME_SPAN", "PRONOUNS", "_NEGATION", "_HEDGE"):
        assert banned not in names


def test_labels_never_claim_more_than_code_established():
    a = msg("a", "user", "Sol: I fence.")
    data = run(window(a), {"T1": slot(mem("Sol fences.", "I fence", "Sol"))}).memories[0].to_dict()
    assert data["semantic_support"] == SEMANTIC_NOT_VERIFIED
    assert not {"confirmed", "verified", "true", "correct"} & set(data)


# ---- sparse output and code-side normalization ------------------------------

SPARSE_DEFAULTS = {"reason": "", "reason_quote": "", "time_expression": "", "time_value": None, "context": [], "uncertain": ""}
DOCK = msg("a", "user", "Gwen: I chose oak for the dock because it lasts longer in 2030.")
DOCK_FIELDS = {"statement": "Gwen chose oak for the dock.", "subject": "Gwen", "subject_status": "stated", "predicate": "chose",
               "object": "oak", "object_status": "stated", "category": "decision", "statement_type": "assertion",
               "time_kind": "current", "quote": "I chose oak for the dock"}


def normalized_memory(item):
    result = run(window(DOCK), {"T1": slot(item)})
    return result, result.normalized["targets"]["T1"]["memories"]


def test_all_fields_returned_are_kept_exactly():
    earlier = msg("z", "user", "Gwen: Which wood lasts?", 0)
    item = {**DOCK_FIELDS, "reason": "It lasts longer.", "reason_quote": "because it lasts longer", "time_expression": "in 2030", "time_value": "2030",
            "uncertain": "May be a provisional choice.", "context": [{"ref": "C1", "quote": "Which wood lasts?"}]}
    result = run(window(DOCK, context=[earlier]), {"T1": slot(item)})
    assert result.normalized["targets"]["T1"]["memories"][0] == item
    m = result.memories[0]
    assert (m.reason, m.reason_quote, m.temporal.expression, m.temporal.normalized, m.model_uncertainty) == (
        "It lasts longer.", "because it lasts longer", "in 2030", "2030", "May be a provisional choice.")
    assert m.message_ids == ("a", "z")


def test_some_optional_fields_omitted_keeps_what_was_returned():
    item = {**DOCK_FIELDS, "reason": "It lasts longer.", "reason_quote": "because it lasts longer"}
    result, normalized = normalized_memory(item)
    assert normalized[0] == {**SPARSE_DEFAULTS, **item}
    assert result.memories[0].reason == "It lasts longer."
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED


def test_all_optional_fields_omitted_get_defaults_and_are_not_flagged():
    result, normalized = normalized_memory(dict(DOCK_FIELDS))
    assert normalized[0] == {**SPARSE_DEFAULTS, **DOCK_FIELDS}
    assert result.issues == []
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED


def test_slot_with_only_an_outcome_is_complete():
    result = validate_targets(json.dumps({"entities": [], "targets": {"T1": {"outcome": "nothing_durable"}}}), window(DOCK))
    assert result.normalized["targets"]["T1"] == {"outcome": "nothing_durable", "note": "", "memories": [], "relationships": []}
    assert result.status == "ok"


@pytest.mark.parametrize(("field", "empty"), [("context", []), ("context", None), ("reason", None), ("reason", ""), ("time_value", None), ("uncertain", None)])
def test_empty_and_null_values_mean_absent(field, empty):
    result, normalized = normalized_memory({**DOCK_FIELDS, field: empty})
    assert normalized[0] == {**SPARSE_DEFAULTS, **DOCK_FIELDS}
    assert result.memories[0].status == STATUS_EVIDENCE_VERIFIED


def test_null_lists_in_a_slot_are_empty():
    result = validate_targets(json.dumps({"entities": [], "targets": {"T1": {"outcome": "question", "memories": None, "relationships": None, "note": None}}}), window(DOCK))
    assert result.status == "ok"
    assert result.normalized["targets"]["T1"]["memories"] == []


@pytest.mark.parametrize("key", ["speaker", "confidence", "memory_id", "superseded"])
def test_unexpected_key_rejects_the_record_and_keeps_its_neighbours(key):
    result = run(window(DOCK), {"T1": slot({**DOCK_FIELDS, key: "x"}, {**DOCK_FIELDS, "statement": "Gwen picked oak."})})
    assert [m.statement for m in result.memories] == ["Gwen picked oak."]
    assert codes(result) == ["UNEXPECTED_KEY"]
    assert key in result.issues[0].detail
    assert "Gwen chose oak" in result.issues[0].detail


def test_unexpected_key_in_a_citation_slot_entity_and_response():
    bad_cite = {**DOCK_FIELDS, "context": [{"ref": "T1", "quote": "oak", "extra": 1}]}
    assert codes(run(window(DOCK), {"T1": slot(bad_cite)})) == ["UNEXPECTED_KEY"]
    bad_slot = {"outcome": "extracted", "rogue": 1}
    assert {"UNEXPECTED_KEY", "MISSING_TARGET_OUTCOME"} <= set(codes(run(window(DOCK), {"T1": bad_slot})))
    assert codes(run(window(DOCK), {"T1": slot(outcome="nothing_durable")}, entities=[{"name": "Gwen", "type": "person", "rogue": 1}])) == ["UNEXPECTED_KEY"]
    top = validate_targets(json.dumps({"entities": [], "targets": {"T1": {"outcome": "question"}}, "extra": 1}), window(DOCK))
    assert (top.status, top.issues[0].code) == ("error", "UNEXPECTED_KEY")


@pytest.mark.parametrize(("field", "value"), [("reason", 5), ("uncertain", False), ("time_value", 2030), ("context", "C1"), ("statement", ["x"]), ("quote", 1)])
def test_wrong_value_type_rejects_the_record(field, value):
    result = run(window(DOCK), {"T1": slot({**DOCK_FIELDS, field: value})})
    assert result.memories == []
    assert codes(result) == ["MALFORMED_FIELD"]


def test_wrong_type_in_a_slot_or_entity_is_rejected():
    assert "MALFORMED_FIELD" in codes(run(window(DOCK), {"T1": {"outcome": "extracted", "memories": "lots"}}))
    assert "MALFORMED_FIELD" in codes(run(window(DOCK), {"T1": slot(outcome="question")}, entities=[{"name": "Gwen", "type": "person", "aliases": "G"}]))


def test_normalization_never_overwrites_or_mutates_returned_values():
    item = {**DOCK_FIELDS, "reason": "Lasts longer.", "reason_quote": "it lasts longer"}
    original = json.loads(json.dumps(item))
    issues = []
    out = normalize_response({"entities": [], "targets": {"T1": {"outcome": "extracted", "memories": [item]}}}, issues)
    memory = out["targets"]["T1"]["memories"][0]
    assert all(memory[k] == v for k, v in original.items())
    assert item == original
    assert issues == []


@pytest.mark.parametrize("extra", [{}, {"reason": "Lasts longer.", "reason_quote": "it lasts longer"}, {"time_expression": "in 2030"}])
def test_normalized_output_always_has_the_complete_structure(extra):
    rel = {"subject": "Gwen", "type": "chose", "object": "oak", "statement_type": "assertion", "time_kind": "current", "quote": "I chose oak"}
    data = {"entities": [{"name": "Gwen", "type": "person"}], "targets": {"T1": {"outcome": "extracted", "memories": [{**DOCK_FIELDS, **extra}], "relationships": [rel]}}}
    out = validate_targets(json.dumps(data), window(DOCK)).normalized
    assert set(out) == {"entities", "targets"}
    assert set(out["entities"][0]) == {"name", "type", "aliases"}
    assert set(out["targets"]["T1"]) == {"outcome", "note", "memories", "relationships"}
    assert set(out["targets"]["T1"]["memories"][0]) == set(MEMORY_FIELDS)
    assert set(out["targets"]["T1"]["relationships"][0]) == set(RELATIONSHIP_FIELDS)


def test_schema_allows_optional_fields_and_forbids_other_keys_at_every_level():
    schema = targets_output_schema(window(DOCK))
    slot_schema = schema["properties"]["targets"]["properties"]["T1"]
    memory = slot_schema["properties"]["memories"]["items"]
    relation = slot_schema["properties"]["relationships"]["items"]
    entity = schema["properties"]["entities"]["items"]
    cite = memory["properties"]["context"]["items"]
    for obj in (schema, schema["properties"]["targets"], slot_schema, memory, relation, entity, cite):
        assert obj["additionalProperties"] is False
    assert slot_schema["required"] == ["outcome"]
    assert not set(SPARSE_DEFAULTS) & set(memory["required"])
    assert not {"context", "uncertain", "time_expression", "time_value"} & set(relation["required"])
    assert set(memory["properties"]) == set(MEMORY_FIELDS)
    assert set(relation["properties"]) == set(RELATIONSHIP_FIELDS)
