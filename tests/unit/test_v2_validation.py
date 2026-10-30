import json

import pytest

from continuator_memory.models import Message
from continuator_memory.v2.models import (
    STATUS_FAILED,
    STATUS_MISSING,
    STATUS_NO_DURABLE,
    STATUS_REJECTED,
    SUPPORT_IDENTITY,
    SUPPORT_SOURCE,
    Claim,
    Provenance,
)
from continuator_memory.v2.validation import FLAGGED, REJECTED, validate_atomization
from continuator_memory.v2.windowing import ContextWindow


def window(targets, context=(), timestamps=None):
    timestamps = timestamps or {}
    msgs = [Message(mid, "conv", i, role, content, timestamps.get(mid))
            for i, (mid, role, content) in enumerate([*context, *targets])]
    return ContextWindow("conv", 0, tuple(msgs[:len(context)]), tuple(msgs[len(context):]))


def claim(cid, content, entities=(), qualifiers=(), temporal=None, polarity="positive", source=None):
    return {"claim_id": cid, "content": content, "entities": list(entities), "qualifiers": list(qualifiers),
            "temporal": temporal or {"expression": "", "resolved_date": ""}, "polarity": polarity,
            "claim_type": "", "source_message_id": source or cid.rsplit(":c", 1)[0]}


def rel(src, dst, rtype, evidence="", source=None):
    return {"from_claim_id": src, "to_claim_id": dst, "type": rtype,
            "source_message_id": source or src.rsplit(":c", 1)[0], "evidence": evidence}


def entry(mid, claims=(), relations=(), status="claims"):
    return {"source_message_id": mid, "status": status, "claims": list(claims), "explicit_relations": list(relations)}


def run(w, entries, prior=()):
    return validate_atomization(json.dumps({"messages": list(entries)}), w, prior)


def codes(result, action=None):
    return [i.code for i in result.issues if action is None or i.action == action]


def prior_claim(cid, content, entities=(), position=-1):
    return Claim(cid, content, Provenance("conv", cid.rsplit(":c", 1)[0], position, "user"), tuple(entities))


# ---- schema -----------------------------------------------------------------


@pytest.mark.parametrize("raw,code", [
    ("not json", "INVALID_JSON"),
    ("[]", "INVALID_SCHEMA"),
    ('{"claims": []}', "INVALID_SCHEMA"),
    (None, "INVALID_JSON"),
])
def test_malformed_output_fails_the_whole_window(raw, code):
    w = window([("m1", "user", "I live in Pune.")])
    result = validate_atomization(raw, w)
    assert result.error and codes(result) == [code]
    assert [m.status for m in result.messages] == [STATUS_FAILED]
    assert result.claims == []


def test_non_list_claims_rejected():
    w = window([("m1", "user", "I live in Pune.")])
    result = validate_atomization(json.dumps({"messages": [
        {"source_message_id": "m1", "status": "claims", "claims": "x", "explicit_relations": []}]}), w)
    assert "INVALID_SCHEMA" in codes(result, REJECTED)


def test_missing_target_entry_is_flagged_not_silent():
    w = window([("m1", "user", "I live in Pune."), ("m2", "user", "I have a cat.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The user lives in Pune.")])])
    assert result.messages[1].status == STATUS_MISSING
    assert "MISSING_MESSAGE" in codes(result, FLAGGED)


def test_claims_status_with_empty_list_is_flagged():
    w = window([("m1", "user", "I live in Pune.")])
    result = run(w, [entry("m1", [])])
    assert "EMPTY_CLAIMS_STATUS" in codes(result, FLAGGED)


def test_no_durable_assertion_accepted():
    w = window([("m1", "assistant", "Great, anything else?")])
    result = run(w, [entry("m1", status="no_durable_assertion")])
    assert result.messages[0].status == STATUS_NO_DURABLE and result.issues == []


# ---- claims -----------------------------------------------------------------


def test_claim_ids_are_canonical_and_deterministic():
    w = window([("m1", "user", "I live in Pune and I have two cats.")])
    doc = [entry("m1", [claim("x7", "The user lives in Pune.", source="m1"),
                        claim("weird", "The user has two cats.", qualifiers=["two"], source="m1")])]
    first, second = run(w, doc), run(w, doc)
    assert [c.claim_id for c in first.claims] == ["m1:c1", "m1:c2"]
    assert [c.to_dict() for c in first.claims] == [c.to_dict() for c in second.claims]


def test_multiple_claims_from_one_message_preserved():
    text = "I work in Delhi, I have two cats, and I run every Sunday."
    w = window([("m1", "user", text)])
    result = run(w, [entry("m1", [claim("m1:c1", "The user works in Delhi.", ["Delhi"]),
                                  claim("m1:c2", "The user has two cats.", qualifiers=["two"]),
                                  claim("m1:c3", "The user runs every Sunday.", qualifiers=["every Sunday"])])])
    assert [c.content for c in result.claims] == [
        "The user works in Delhi.", "The user has two cats.", "The user runs every Sunday."]
    assert result.issues == []


def test_empty_claim_rejected():
    w = window([("m1", "user", "I live in Pune.")])
    result = run(w, [entry("m1", [claim("m1:c1", "  "), claim("m1:c2", "The user lives in Pune.")])])
    assert "EMPTY_CLAIM" in codes(result, REJECTED)
    assert [c.claim_id for c in result.claims] == ["m1:c1"]
    assert result.claims[0].content == "The user lives in Pune."


def test_duplicate_claim_id_rejected():
    w = window([("m1", "user", "I live in Pune and work at Infosys.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The user lives in Pune."),
                                  claim("m1:c1", "The user works at Infosys.")])])
    assert "DUPLICATE_CLAIM_ID" in codes(result, REJECTED)
    assert len(result.claims) == 1


def test_invalid_source_message_rejected():
    w = window([("m1", "user", "I live in Pune.")])
    result = run(w, [entry("m9", [claim("m9:c1", "The user lives in Pune.")]),
                     entry("m1", [claim("m1:c1", "The user lives in Pune.", source="m42")])])
    assert codes(result, REJECTED).count("INVALID_SOURCE_MESSAGE") == 2
    assert result.messages[0].status == STATUS_REJECTED


def test_claim_attributed_to_context_message_is_impossible():
    w = window([("m2", "user", "Also, it has a rooftop pool.")], context=[("m1", "user", "We booked Hotel Lumo.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The user booked Hotel Lumo.")]),
                     entry("m2", [claim("m2:c1", "The user booked Hotel Lumo.", source="m1")])])
    assert codes(result, REJECTED) == ["IMPOSSIBLE_SOURCE", "IMPOSSIBLE_SOURCE"]


def test_unsupported_content_rejected():
    w = window([("m1", "user", "I live in Pune.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The user lives in Pune and works at Infosys.")])])
    assert "UNSUPPORTED_CLAIM" in codes(result, REJECTED)
    assert result.claims == []


def test_source_provenance_is_attached():
    w = window([("m1", "user", "I live in Pune.")], timestamps={"m1": "2026-03-01T10:00:00"})
    c = run(w, [entry("m1", [claim("m1:c1", "The user lives in Pune.")])]).claims[0]
    assert (c.source_message_id, c.provenance.role, c.provenance.position, c.provenance.timestamp,
            c.provenance.conversation_id) == ("m1", "user", 0, "2026-03-01T10:00:00", "conv")
    assert c.support == SUPPORT_SOURCE


# ---- prior state: identity, not evidence ------------------------------------


def test_prior_state_may_resolve_identity():
    w = window([("m5", "user", "She moved to Bangalore.")])
    prior = [prior_claim("m1:c1", "Leah is the person being discussed.", ["Leah"])]
    result = run(w, [entry("m5", [claim("m5:c1", "Leah moved to Bangalore.", ["Leah", "Bangalore"])])], prior)
    assert result.issues == []
    c = result.claims[0]
    assert c.content == "Leah moved to Bangalore."
    assert c.support == SUPPORT_IDENTITY and c.resolved_from == ("m1:c1",)


def test_prior_state_may_not_supply_a_new_fact():
    w = window([("m5", "user", "She moved to Bangalore.")])
    prior = [prior_claim("m1:c1", "Leah is the person being discussed.", ["Leah"])]
    result = run(w, [entry("m5", [claim("m5:c1", "Leah got a new job in Bangalore.", ["Leah", "Bangalore"])])], prior)
    assert "UNSUPPORTED_CLAIM" in codes(result, REJECTED)
    assert result.claims == []


def test_prior_fact_copied_into_new_claim_rejected():
    w = window([("m5", "user", "She moved to Bangalore.")])
    prior = [prior_claim("m1:c1", "Leah got a new job at Wipro.", ["Leah", "Wipro"])]
    result = run(w, [entry("m5", [claim("m5:c1", "Leah moved to Bangalore for her new job at Wipro.",
                                        ["Leah", "Bangalore", "Wipro"])])], prior)
    assert "UNSUPPORTED_CLAIM" in codes(result, REJECTED)


def test_pronoun_resolved_from_context_message():
    w = window([("m3", "user", "It's on October 12th at 10am.")],
               context=[("m1", "user", "I have a job interview with Lumen Health."),
                        ("m2", "assistant", "When is the interview?")])
    result = run(w, [entry("m3", [claim("m3:c1", "The interview is on October 12th at 10am.",
                                        temporal={"expression": "October 12th at 10am", "resolved_date": ""})])])
    assert result.issues == []
    assert result.claims[0].support == SUPPORT_IDENTITY
    assert result.claims[0].resolved_from == ("m2",)


def test_name_without_referring_expression_is_not_resolved():
    w = window([("m2", "user", "Moved to Bangalore last week.")], context=[("m1", "user", "Leah is my sister.")])
    result = run(w, [entry("m2", [claim("m2:c1", "Leah moved to Bangalore last week.", ["Leah"])])])
    assert "UNSUPPORTED_CLAIM" in codes(result, REJECTED)


# ---- annotations ------------------------------------------------------------


def test_temporal_metadata_preserved():
    w = window([("m1", "user", "I'm visiting Spain next month.")])
    c = run(w, [entry("m1", [claim("m1:c1", "The user is visiting Spain next month.",
                                   temporal={"expression": "next month", "resolved_date": ""})])]).claims[0]
    assert c.temporal.expression == "next month" and c.temporal.resolved_date is None


def test_resolved_date_requires_a_timestamp():
    text = "My dentist appointment is tomorrow."
    doc = [entry("m1", [claim("m1:c1", "The user's dentist appointment is tomorrow.",
                              temporal={"expression": "tomorrow", "resolved_date": "2026-03-02"})])]
    without = run(window([("m1", "user", text)]), doc)
    assert without.claims[0].temporal.resolved_date is None
    assert "RESOLVED_WITHOUT_TIMESTAMP" in codes(without, FLAGGED)
    with_ts = run(window([("m1", "user", text)], timestamps={"m1": "2026-03-01"}), doc)
    assert with_ts.claims[0].temporal.resolved_date == "2026-03-02" and with_ts.issues == []


@pytest.mark.parametrize("temporal,code", [
    ("tomorrow", "MALFORMED_TEMPORAL"),
    ({"expression": "", "resolved_date": "2026-03-02"}, "MALFORMED_TEMPORAL"),
    ({"expression": "tomorrow", "resolved_date": "March 2"}, "MALFORMED_TEMPORAL"),
    ({"expression": "on March 2nd", "resolved_date": ""}, "TEMPORAL_NOT_IN_SOURCE"),
])
def test_malformed_temporal_flagged(temporal, code):
    w = window([("m1", "user", "My dentist appointment is tomorrow.")])
    item = claim("m1:c1", "The user's dentist appointment is tomorrow.")
    item["temporal"] = temporal
    result = run(w, [entry("m1", [item])])
    assert code in codes(result, FLAGGED)
    assert len(result.claims) == 1


def test_negation_preserved():
    w = window([("m1", "user", "I don't want to move to London.")])
    c = run(w, [entry("m1", [claim("m1:c1", "The user doesn't want to move to London.", ["London"],
                                   polarity="negative")])]).claims[0]
    assert c.content == "The user doesn't want to move to London." and c.polarity == "negative"


def test_invalid_polarity_flagged():
    w = window([("m1", "user", "I live in Pune.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The user lives in Pune.", polarity="true")])])
    assert "INVALID_POLARITY" in codes(result, FLAGGED) and result.claims[0].polarity is None


def test_unsupported_qualifier_and_entity_dropped():
    w = window([("m1", "user", "I live in Pune.")])
    c_doc = claim("m1:c1", "The user lives in Pune.", entities=["Pune", "Mumbai"], qualifiers=["since 2019"])
    result = run(w, [entry("m1", [c_doc])])
    c = result.claims[0]
    assert c.entities == ("Pune",) and c.qualifiers == ()
    assert {"UNSUPPORTED_ENTITY", "UNSUPPORTED_QUALIFIER"} <= set(codes(result, FLAGGED))


# ---- relations --------------------------------------------------------------


def test_explicit_causal_relation_accepted_with_deterministic_ids():
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")])
    doc = [entry("m1", [claim("m1:c1", "The launch moved to July."), claim("m1:c2", "The review found bugs.")],
                 [rel("m1:c1", "m1:c2", "CAUSED_BY", "because")])]
    first, second = run(w, doc), run(w, doc)
    assert first.issues == []
    r = first.relations[0]
    assert (r.relation_id, r.from_claim_id, r.to_claim_id, r.type, r.evidence) == (
        "m1:r1", "m1:c1", "m1:c2", "CAUSED_BY", "because")
    assert r.evidence_span == (25, 32)
    assert [x.to_dict() for x in first.relations] == [x.to_dict() for x in second.relations]


def test_no_causal_relation_for_simple_and():
    w = window([("m1", "user", "Ines prefers the seaside and Omar gets seasick.")])
    result = run(w, [entry("m1", [claim("m1:c1", "Ines prefers the seaside.", ["Ines"]),
                                  claim("m1:c2", "Omar gets seasick.", ["Omar"])],
                           [rel("m1:c2", "m1:c1", "CAUSED_BY")])])
    assert "UNSUPPORTED_RELATION" in codes(result, REJECTED)
    assert result.relations == [] and len(result.claims) == 2


def test_explicit_because_relation_accepted():
    w = window([("m1", "user", "Lumen Health moved the interview to October 14th because one panelist was sick.")])
    result = run(w, [entry("m1", [claim("m1:c1", "Lumen Health moved the interview to October 14th.", ["Lumen Health"]),
                                  claim("m1:c2", "One panelist was sick.")],
                           [rel("m1:c1", "m1:c2", "CAUSED_BY", "because")])])
    assert result.issues == [] and [(r.from_claim_id, r.to_claim_id) for r in result.relations] == [("m1:c1", "m1:c2")]


def test_explicit_so_relation_accepted():
    w = window([("m1", "user", "Marco is going on parental leave, so Hana takes over as team lead starting May 1st.")])
    result = run(w, [entry("m1", [claim("m1:c1", "Marco is going on parental leave.", ["Marco"]),
                                  claim("m1:c2", "Hana takes over as team lead starting May 1st.", ["Hana"])],
                           [rel("m1:c2", "m1:c1", "CAUSED_BY", "so")])])
    assert result.issues == [] and [(r.from_claim_id, r.to_claim_id) for r in result.relations] == [("m1:c2", "m1:c1")]


def test_unrelated_and_facts_rejected_as_causal_even_with_evidence():
    w = window([("m1", "user", "Ines prefers the seaside and Omar gets seasick.")])
    result = run(w, [entry("m1", [claim("m1:c1", "Ines prefers the seaside.", ["Ines"]),
                                  claim("m1:c2", "Omar gets seasick.", ["Omar"])],
                           [rel("m1:c2", "m1:c1", "CAUSED_BY", "and")])])
    assert "UNSUPPORTED_RELATION" in codes(result, REJECTED) and result.relations == []


def test_causal_keyword_elsewhere_does_not_authorize_unrelated_relation():
    text = "For the offsite, Ines prefers somewhere near the sea, and Omar gets seasick, so we should avoid boat trips."
    w = window([("m1", "user", text)])
    claims = [claim("m1:c1", "Ines prefers somewhere near the sea for the offsite.", ["Ines"]),
              claim("m1:c2", "Omar gets seasick.", ["Omar"]),
              claim("m1:c3", "We should avoid boat trips.")]
    wrong = run(w, [entry("m1", claims, [rel("m1:c2", "m1:c1", "CAUSED_BY", "so")])])
    assert "UNSUPPORTED_RELATION" in codes(wrong, REJECTED) and wrong.relations == []
    wrong_effect = run(w, [entry("m1", claims, [rel("m1:c3", "m1:c1", "CAUSED_BY", "so")])])
    assert wrong_effect.relations == []
    right = run(w, [entry("m1", claims, [rel("m1:c3", "m1:c2", "CAUSED_BY", "so")])])
    assert right.issues == [] and [(r.from_claim_id, r.to_claim_id) for r in right.relations] == [("m1:c3", "m1:c2")]


def test_explicit_update_of_prior_claim_accepted():
    w = window([("m9", "user", "Update: the budget is now 55,000 euros.")])
    prior = [prior_claim("m3:c1", "The budget is 40,000 euros.")]
    result = run(w, [entry("m9", [claim("m9:c1", "The budget is now 55,000 euros.")],
                           [rel("m9:c1", "m3:c1", "UPDATES", "now")])], prior)
    assert result.issues == []
    assert [(r.from_claim_id, r.to_claim_id, r.type) for r in result.relations] == [("m9:c1", "m3:c1", "UPDATES")]


def test_update_pointing_forward_in_time_rejected():
    w = window([("m9", "user", "Update: the budget is now 55,000 euros.")])
    prior = [prior_claim("m3:c1", "The budget is 40,000 euros.")]
    result = run(w, [entry("m9", [claim("m9:c1", "The budget is now 55,000 euros.")],
                           [rel("m3:c1", "m9:c1", "UPDATES", source="m9")])], prior)
    assert "UNSUPPORTED_DIRECTION" in codes(result, REJECTED)


def test_update_without_change_wording_is_flagged():
    w = window([("m9", "user", "The budget is 55,000 euros.")])
    prior = [prior_claim("m3:c1", "The budget is 40,000 euros.")]
    result = run(w, [entry("m9", [claim("m9:c1", "The budget is 55,000 euros.")],
                           [rel("m9:c1", "m3:c1", "UPDATES", "55,000 euros")])], prior)
    assert "NO_CHANGE_CUE" in codes(result, FLAGGED) and len(result.relations) == 1


@pytest.mark.parametrize("relation,code", [
    (rel("m1:c1", "m1:c9", "CAUSED_BY", "because"), "NONEXISTENT_CLAIM"),
    (rel("m1:c1", "m0:c1", "UPDATES"), "UNAVAILABLE_SOURCE"),
    (rel("m1:c1", "m7:c1", "UPDATES"), "UNAVAILABLE_SOURCE"),
    (rel("m1:c1", "m1:c1", "CAUSED_BY", "because"), "SELF_LOOP"),
    (rel("m1:c1", "m1:c2", "SEMANTICALLY_RELATED"), "INVALID_RELATION_TYPE"),
    (rel("m1:c1", "m1:c2", "CAUSED_BY", source="m0"), "IMPOSSIBLE_SOURCE"),
])
def test_invalid_relations_rejected(relation, code):
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")],
               context=[("m0", "user", "The launch is in June.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The launch moved to July."),
                                  claim("m1:c2", "The review found bugs.")], [relation])])
    assert code in codes(result, REJECTED)
    assert result.relations == []


def test_relation_to_rejected_claim_rejected():
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The launch moved to July."),
                                  claim("m1:c2", "The CEO was angry.")],
                           [rel("m1:c1", "m1:c2", "CAUSED_BY", "because")])])
    assert {"UNSUPPORTED_CLAIM", "NONEXISTENT_CLAIM"} <= set(codes(result, REJECTED))
    assert result.relations == []


def test_relation_ids_follow_model_ids_after_a_rejected_claim():
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")])
    result = run(w, [entry("m1", [claim("m1:c1", ""), claim("m1:c2", "The launch moved to July."),
                                  claim("m1:c3", "The review found bugs.")],
                           [rel("m1:c2", "m1:c3", "CAUSED_BY", "because")])])
    assert [c.claim_id for c in result.claims] == ["m1:c1", "m1:c2"]
    assert [(r.from_claim_id, r.to_claim_id) for r in result.relations] == [("m1:c1", "m1:c2")]


def test_reverse_duplicate_causal_relation_rejected():
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The launch moved to July."), claim("m1:c2", "The review found bugs.")],
                           [rel("m1:c1", "m1:c2", "CAUSED_BY", "because"),
                            rel("m1:c2", "m1:c1", "CAUSED_BY", "because")])])
    assert "UNSUPPORTED_DIRECTION" in codes(result, REJECTED) and len(result.relations) == 1


@pytest.mark.parametrize("evidence", ["as a consequence", "Because", "", "the launch moved to July"])
def test_relation_evidence_must_be_exact_substring(evidence):
    w = window([("m1", "user", "The launch moved to July because the review found bugs.")])
    result = run(w, [entry("m1", [claim("m1:c1", "The launch moved to July."), claim("m1:c2", "The review found bugs.")],
                           [rel("m1:c1", "m1:c2", "CAUSED_BY", evidence)])])
    assert "EVIDENCE_NOT_IN_SOURCE" in codes(result, REJECTED)
    assert result.relations == []


def test_supports_relation_with_paraphrased_evidence_rejected():
    w = window([("m3", "user", "She's allergic to nuts."),
                ("m15", "user", "Omar's birthday falls during the offsite, so let's plan a small cake with no nuts.")],
               context=[("m1", "user", "Ines is our designer.")])
    result = run(w, [entry("m3", [claim("m3:c1", "Ines is allergic to nuts.", ["Ines"])]),
                     entry("m15", [claim("m15:c1", "Let's plan a small cake with no nuts.")],
                           [rel("m15:c1", "m3:c1", "SUPPORTS", "Let's plan a small cake with no nuts for the offsite.")])])
    assert "EVIDENCE_NOT_IN_SOURCE" in codes(result, REJECTED) and result.relations == []


def test_relation_endpoint_from_another_conversation_rejected():
    w = window([("m9", "user", "Update: the budget is now 55,000 euros.")])
    foreign = Claim("m3:c1", "The budget is 40,000 euros.", Provenance("other", "m3", -1, "user"))
    result = run(w, [entry("m9", [claim("m9:c1", "The budget is now 55,000 euros.")],
                           [rel("m9:c1", "m3:c1", "UPDATES", "now")])], [foreign])
    assert "ENDPOINT_OUTSIDE_WINDOW" in codes(result, REJECTED) and result.relations == []
