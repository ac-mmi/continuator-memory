import random

from continuator_memory.v2.consolidation import (
    Consolidator,
    claim_subject,
    consolidate,
    entity_key,
    restatement_key,
)
from continuator_memory.v2.memory import (
    CONTRADICTION,
    CREATE,
    HISTORICAL,
    IGNORE,
    KEEP,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    SUPERSEDE,
    UPDATE,
)
from continuator_memory.v2.models import Claim, Provenance, Relation, Temporal

CONV = "conv"


def claim(cid, content, position, entities=(), claim_type=None, polarity="positive", temporal=None,
          resolved=None, qualifiers=(), metadata=None, ts=None, conv=CONV):
    mid = cid.rsplit(":c", 1)[0]
    return Claim(cid, content, Provenance(conv, mid, position, "user", ts), tuple(entities), tuple(qualifiers),
                 Temporal(temporal, resolved) if temporal else None, polarity, claim_type, metadata=metadata or {})


def rel(src, dst, rtype, n=1):
    mid = src.rsplit(":c", 1)[0]
    return Relation(f"{mid}:r{n}", src, dst, rtype, mid, "evidence")


def statuses(c: Consolidator) -> dict[str, str]:
    return {m.content: m.status for m in c.memories.values()}


def actions(c: Consolidator) -> list[str]:
    return [d.action for d in c.decisions]


# ---- identity helpers -------------------------------------------------------


def test_entity_key_is_exact_normalization_only():
    assert entity_key("The Museum") == "museum"
    assert entity_key("I") == entity_key("the user") == "user"
    assert entity_key("Café Lumière") == "cafe lumiere"
    assert entity_key("Sam") != entity_key("Samantha")


def test_claim_subject_is_deterministic_and_conservative():
    assert claim_subject(claim("a:c1", "The user lives in Delhi.", 0)) == "user"
    assert claim_subject(claim("a:c2", "Sam has a cat.", 0, ["Sam", "cat"])) == "sam"
    assert claim_subject(claim("a:c3", "The museum is closed Monday.", 0, ["museum"])) == "museum"
    # possessive: about the sister, whose identity is unknown
    assert claim_subject(claim("a:c4", "The user's sister lives in Madrid.", 0, ["Madrid"])) is None
    assert claim_subject(claim("a:c5", "She moved to Bangalore.", 0, ["Bangalore"])) is None
    assert claim_subject(claim("a:c6", "Hana leads it.", 0, metadata={"subject": "the user's team"})) == "user s team"


def test_restatement_key_refuses_relative_time_and_unresolved_references():
    assert restatement_key(claim("a:c1", "The user has a dentist appointment tomorrow.", 0)) is None
    assert restatement_key(claim("a:c2", "She moved to Bangalore.", 0)) is None
    assert restatement_key(claim("a:c3", "The user visited Rome.", 0, temporal="last summer")) is None
    assert restatement_key(claim("a:c4", "The user visited Rome in 2019.", 0, temporal="in 2019")) is not None


# ---- CREATE / KEEP ------------------------------------------------------------


def test_create_new_memory_with_provenance():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, ts="2026-01-05T10:00:00")])
    [m] = c.memories.values()
    assert (m.memory_id, m.status, m.claim_id, m.message_id) == ("mem:m1:c1", STATUS_CURRENT, "m1:c1", "m1")
    assert m.created_at == "2026-01-05T10:00:00"
    assert m.valid_from is None  # message timestamp is observation time, not an event date
    assert actions(c) == [CREATE]
    assert [(s.claim_id, s.message_id, s.observed_at) for s in c.sources_of(m.memory_id)] == [
        ("m1:c1", "m1", "2026-01-05T10:00:00")]


def test_D_exact_duplicate_is_one_memory_with_multiple_sources():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, claim_type="residence"),
                           claim("m5:c1", "The user lives in Delhi.", 4, claim_type="residence")])
    assert actions(c) == [CREATE, KEEP]
    assert len(c.memories) == 1 and len(c.current()) == 1
    assert [s.message_id for s in c.sources_of("mem:m1:c1")] == ["m1", "m5"]
    assert c.memory_for_claim("m5:c1").memory_id == "mem:m1:c1"


def test_restatement_ignores_case_and_punctuation_but_not_polarity():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0),
                           claim("m2:c1", "the user lives in delhi", 1),
                           claim("m3:c1", "The user does not live in Delhi.", 2, polarity="negative")])
    assert actions(c) == [CREATE, KEEP, CREATE]
    assert len(c.memories) == 2


def test_reapplying_a_claim_is_ignored_and_foreign_conversation_rejected():
    c = Consolidator(CONV)
    first = claim("m1:c1", "The user lives in Delhi.", 0)
    c.apply([first])
    result = c.apply([first, claim("x1:c1", "The user lives in Rome.", 0, conv="other")])
    assert [d.action for d in result.decisions] == [IGNORE, IGNORE]
    assert result.ignored == ["m1:c1", "x1:c1"]
    assert len(c.memories) == 1 and len(c.sources) == 1


def test_same_wording_with_relative_time_is_not_merged():
    c = consolidate(CONV, [claim("m1:c1", "The user has a dentist appointment tomorrow.", 0, ts="2026-01-01"),
                           claim("m9:c1", "The user has a dentist appointment tomorrow.", 8, ts="2026-03-01")])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2


# ---- versioning: positive fixtures ------------------------------------------------------------


def test_A_move_supersedes_residence_and_preserves_history():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, ["Delhi"], "residence"),
                           claim("m4:c1", "The user moved to Bangalore.", 3, ["Bangalore"], "residence")])
    assert actions(c) == [CREATE, SUPERSEDE]
    old, new = c.memories["mem:m1:c1"], c.memories["mem:m4:c1"]
    assert (old.status, old.superseded_by) == (STATUS_SUPERSEDED, new.memory_id)
    assert new.status == STATUS_CURRENT
    assert [m.content for m in c.current()] == ["The user moved to Bangalore."]
    assert [(l.newer_memory_id, l.older_memory_id, l.type) for l in c.links] == [
        ("mem:m4:c1", "mem:m1:c1", "SUPERSEDES")]
    assert [m.memory_id for m in c.history(new.memory_id)] == ["mem:m1:c1"]
    # the old source remains queryable
    assert [s.claim_id for s in c.sources_of(old.memory_id)] == ["m1:c1"]


def test_B_team_lead_handover_versioned_when_metadata_identifies_the_slot():
    subject = {"subject": "the user's team lead"}
    c = consolidate(CONV, [
        claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"], "team lead", metadata=subject),
        claim("m2:c1", "Hana is taking over as the user's team lead.", 1, ["Hana"], "team lead", metadata=subject)])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert statuses(c) == {"The user's team lead is Marco.": STATUS_SUPERSEDED,
                           "Hana is taking over as the user's team lead.": STATUS_CURRENT}


def test_B_team_lead_without_identity_metadata_preserves_both():
    c = consolidate(CONV, [
        claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"], "team lead"),
        claim("m2:c1", "Hana is taking over as the user's team lead.", 1, ["Hana"], "team lead")])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2 and c.links == []


def test_B_team_lead_with_explicit_updates_relation():
    c = consolidate(CONV, [
        claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"]),
        claim("m2:c1", "Hana is taking over as the user's team lead.", 1, ["Hana"])],
        [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE]
    assert c.memories["mem:m1:c1"].status == STATUS_SUPERSEDED


def test_C_preference_change_is_versioned():
    c = consolidate(CONV, [
        claim("m1:c1", "The user prefers Thursday meetings.", 0, claim_type="meeting preference"),
        claim("m3:c1", "The user prefers Friday meetings now.", 2, claim_type="Meeting preference")])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert [m.content for m in c.current()] == ["The user prefers Friday meetings now."]


def test_E_explicit_updates_relation_supersedes_deterministically():
    c = consolidate(CONV, [claim("m1:c2", "Leah lives in Madrid.", 0, ["Leah", "Madrid"]),
                           claim("m4:c1", "Leah actually lives in Valencia now.", 3, ["Leah", "Valencia"])],
                    [rel("m4:c1", "m1:c2", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE]
    d = c.decisions[1]
    assert d.superseded == ("mem:m1:c2",) and d.unresolved is None
    assert [(l.type, l.relation_id) for l in c.links] == [("UPDATES", "m4:r1")]
    assert statuses(c) == {"Leah lives in Madrid.": STATUS_SUPERSEDED,
                           "Leah actually lives in Valencia now.": STATUS_CURRENT}


def test_update_chain_keeps_every_version():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, claim_type="residence"),
                           claim("m2:c1", "The user moved to Bangalore.", 1, claim_type="residence"),
                           claim("m3:c1", "The user moved to Pune.", 2, claim_type="residence")])
    assert actions(c) == [CREATE, SUPERSEDE, SUPERSEDE]
    assert [m.content for m in c.current()] == ["The user moved to Pune."]
    assert [m.memory_id for m in c.history("mem:m3:c1")] == ["mem:m2:c1", "mem:m1:c1"]
    assert len(c.memories) == 3 and len(c.sources) == 3


def test_validity_dates_come_only_from_resolved_claim_dates():
    c = consolidate(CONV, [
        claim("m1:c1", "The user lives in Delhi.", 0, claim_type="residence", ts="2026-01-01T09:00:00"),
        claim("m2:c1", "The user moved to Bangalore on March 3.", 1, claim_type="residence",
              temporal="on March 3", resolved="2026-03-03", ts="2026-03-10T09:00:00")])
    old, new = c.memories["mem:m1:c1"], c.memories["mem:m2:c1"]
    assert (old.valid_from, old.valid_until) == (None, "2026-03-03")
    assert (new.valid_from, new.created_at) == ("2026-03-03", "2026-03-10T09:00:00")


def test_change_dated_before_the_current_state_does_not_supersede():
    # Phase 2B: event order shows the later-observed move is the older version.
    c = consolidate(CONV, [
        claim("m1:c1", "The user lives in Pune since May.", 0, claim_type="residence", temporal="since May",
              resolved="2026-05-01"),
        claim("m2:c1", "The user moved to Bangalore in March.", 1, claim_type="residence", temporal="in March",
              resolved="2026-03-01")])
    assert actions(c) == [CREATE, HISTORICAL]
    assert "dated before" in c.decisions[1].reason
    assert [m.content for m in c.current()] == ["The user lives in Pune since May."]
    assert c.memories["mem:m2:c1"].superseded_by == "mem:m1:c1"


def test_future_or_conditional_change_does_not_supersede():
    # Phase 2B: the future claim is kept with status "future", not counted as current.
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, claim_type="residence"),
                           claim("m2:c1", "The user will move to Bangalore now that the job is confirmed.", 1,
                                 claim_type="residence")])
    assert actions(c) == [CREATE, CREATE]
    assert [m.content for m in c.current()] == ["The user lives in Delhi."]
    assert c.memories["mem:m2:c1"].status == "future"


# ---- CONTRADICTION ------------------------------------------------------------


def test_F_explicit_contradiction_preserves_both_without_a_winner():
    c = consolidate(CONV, [claim("m1:c1", "The museum is closed Monday.", 0, ["museum"]),
                           claim("m2:c1", "The museum is not closed Monday.", 1, ["museum"], polarity="negative")],
                    [rel("m2:c1", "m1:c1", "CONTRADICTS")])
    assert actions(c) == [CREATE, CONTRADICTION]
    assert set(statuses(c).values()) == {STATUS_CONTESTED}
    assert len(c.current()) == 2  # contested memories both remain available
    assert [(l.newer_memory_id, l.older_memory_id, l.type) for l in c.links] == [
        ("mem:m2:c1", "mem:m1:c1", "CONTRADICTS")]
    assert c.apply([]).contradictions == []
    assert c.decisions[1].contradicted == ("mem:m1:c1",)


# ---- explicit relations: unsafe or unresolvable --------------------------------


def test_updates_with_unresolved_reference_preserves_both():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah", "Madrid"]),
                           claim("m2:c1", "She moved to Bangalore.", 1, ["Bangalore"])],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, CREATE]
    assert set(statuses(c).values()) == {STATUS_CURRENT}
    assert "unresolved reference in m2:c1" in c.decisions[1].unresolved
    assert c.links == []


def test_updates_with_conflicting_explicit_subjects_preserves_both():
    c = consolidate(CONV, [claim("m1:c1", "Sam has a cat.", 0, ["Sam"], metadata={"subject": "Sam"}),
                           claim("m2:c1", "Alex now has a cat.", 1, ["Alex"], metadata={"subject": "Alex"})],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, CREATE]
    assert "explicit subjects differ" in c.decisions[1].unresolved


def test_updates_to_unknown_or_already_superseded_target_is_unresolved():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0),
                           claim("m2:c1", "The user lives in Bangalore now.", 1),
                           claim("m3:c1", "The user lives in Pune now.", 2),
                           claim("m4:c1", "The user lives in Goa now.", 3)],
                    [rel("m2:c1", "m1:c1", "UPDATES"), rel("m3:c1", "m1:c1", "UPDATES"),
                     rel("m4:c1", "m0:c9", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE, CREATE, CREATE]
    assert "already superseded by mem:m2:c1" in c.decisions[2].unresolved
    assert "not consolidated" in c.decisions[3].unresolved
    assert len(c.memories) == 4  # nothing is lost


def test_updates_of_a_historical_memory_links_without_changing_it():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0),
                           claim("m2:c1", "The user moved to Bangalore.", 1)],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert statuses(c) == {"The user used to live in Delhi.": STATUS_HISTORICAL,
                           "The user moved to Bangalore.": STATUS_CURRENT}
    assert [l.type for l in c.links] == ["UPDATES"]


def test_refers_to_links_without_changing_state_and_causal_relations_are_not_consumed():
    c = consolidate(CONV, [claim("m1:c1", "The user booked a table at Nopa.", 0, ["Nopa"]),
                           claim("m2:c1", "The booking at Nopa is for 8 pm.", 1, ["Nopa"]),
                           claim("m2:c2", "The user chose Nopa because of the reviews.", 1, ["Nopa"])],
                    [rel("m2:c1", "m1:c1", "REFERS_TO", 1), rel("m2:c2", "m1:c1", "CAUSED_BY", 2)])
    assert set(statuses(c).values()) == {STATUS_CURRENT}
    assert [(l.newer_memory_id, l.older_memory_id, l.type) for l in c.links] == [
        ("mem:m2:c1", "mem:m1:c1", "REFERS_TO")]


def test_relation_arriving_after_its_claim_is_reported_not_applied():
    c = Consolidator(CONV)
    c.apply([claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah"]),
             claim("m2:c1", "Leah lives in Valencia now.", 1, ["Leah"])])
    result = c.apply([], [rel("m2:c1", "m1:c1", "UPDATES")])
    assert result.unresolved_relations == [("m2:r1", "m2:c1 was consolidated before the relation arrived")]
    assert len(c.current()) == 2


# ---- HISTORICAL ---------------------------------------------------------------


def test_historical_statement_is_kept_alongside_current_state():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0, ["Delhi"], "residence"),
                           claim("m2:c1", "The user lives in Bangalore now.", 1, ["Bangalore"], "residence")])
    assert actions(c) == [HISTORICAL, CREATE]
    assert statuses(c) == {"The user used to live in Delhi.": STATUS_HISTORICAL,
                           "The user lives in Bangalore now.": STATUS_CURRENT}
    assert c.links == []


def test_used_to_then_moved_preserves_historical_and_current():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0, claim_type="residence"),
                           claim("m2:c1", "The user moved to Bangalore.", 1, claim_type="residence")])
    assert statuses(c) == {"The user used to live in Delhi.": STATUS_HISTORICAL,
                           "The user moved to Bangalore.": STATUS_CURRENT}


def test_restating_a_superseded_value_is_kept_apart_and_flagged():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Delhi.", 0, claim_type="residence"),
                           claim("m2:c1", "The user moved to Bangalore.", 1, claim_type="residence"),
                           claim("m3:c1", "The user lives in Delhi.", 2, claim_type="residence")])
    assert actions(c) == [CREATE, SUPERSEDE, CREATE]
    assert c.memories["mem:m1:c1"].status == STATUS_SUPERSEDED
    assert "restates superseded mem:m1:c1" in c.decisions[2].unresolved


# ---- negative cases: must not merge -------------------------------------------


def test_N1_different_attributes_sharing_a_word():
    c = consolidate(CONV, [claim("m1:c1", "The user likes coffee.", 0, ["coffee"], "preference"),
                           claim("m2:c1", "The user works at Coffee Labs now.", 1, ["Coffee Labs"], "employment")])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2 and c.links == []


def test_N2_estimate_and_final_bill_are_different_events():
    c = consolidate(CONV, [
        claim("m1:c1", "The user estimated the repair at €240.", 0, ["repair"], "repair cost", qualifiers=["€240"]),
        claim("m5:c1", "The final repair bill was €310 including rotors.", 4, ["repair bill"], "repair cost",
              qualifiers=["€310", "including rotors"])])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2 and c.links == []


def test_N2_same_subject_and_type_without_change_wording_does_not_supersede():
    c = consolidate(CONV, [claim("m1:c1", "The user quoted €240 for brake pads.", 0, claim_type="cost"),
                           claim("m2:c1", "The user paid €310 including rotors.", 1, claim_type="cost")])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2


def test_N3_historical_statement_is_not_overwritten():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0),
                           claim("m2:c1", "The user lives in Bangalore now.", 1)])
    assert [m.status for m in c.memories.values()] == [STATUS_HISTORICAL, STATUS_CURRENT]


def test_N4_different_entities_with_the_same_attribute():
    c = consolidate(CONV, [claim("m1:c1", "Sam has a cat.", 0, ["Sam", "cat"], "pet"),
                           claim("m2:c1", "Alex now has a cat.", 1, ["Alex", "cat"], "pet")])
    assert actions(c) == [CREATE, CREATE]
    assert {m.subject for m in c.current()} == {"sam", "alex"}


def test_N5_ambiguous_identity_is_never_merged():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah", "Madrid"], "residence"),
                           claim("m2:c1", "She moved to Bangalore.", 1, ["Bangalore"], "residence"),
                           claim("m3:c1", "She moved to Bangalore.", 2, ["Bangalore"], "residence")])
    assert actions(c) == [CREATE, CREATE, CREATE]
    assert len(c.current()) == 3
    assert all("unresolved reference" in d.unresolved for d in c.decisions[1:])


def test_N6_similar_wording_different_fact():
    c = consolidate(CONV, [claim("m1:c1", "The user's flight is to Madrid.", 0, ["Madrid"], "travel"),
                           claim("m2:c1", "The user's sister lives in Madrid now.", 1, ["Madrid"], "travel")])
    assert actions(c) == [CREATE, CREATE]
    assert len(c.current()) == 2


def test_N7_conflicting_statements_without_relation_are_both_kept():
    c = consolidate(CONV, [claim("m1:c1", "The museum is closed Monday.", 0, ["museum"], "opening hours"),
                           claim("m2:c1", "The museum is closed Tuesday.", 1, ["museum"], "opening hours")])
    assert actions(c) == [CREATE, CREATE]
    assert set(statuses(c).values()) == {STATUS_CURRENT} and c.links == []


def test_several_current_memories_in_a_slot_block_implicit_supersession():
    c = consolidate(CONV, [claim("m1:c1", "The user speaks Spanish.", 0, claim_type="language"),
                           claim("m2:c1", "The user speaks French.", 1, claim_type="language"),
                           claim("m3:c1", "The user now speaks German.", 2, claim_type="language")])
    assert actions(c) == [CREATE, CREATE, CREATE]
    assert "2 current memories share subject and claim_type" in c.decisions[2].unresolved


# ---- invariants ---------------------------------------------------------------


STREAM = [
    claim("m1:c1", "The user lives in Delhi.", 0, ["Delhi"], "residence"),
    claim("m1:c2", "Leah lives in Madrid.", 0, ["Leah", "Madrid"]),
    claim("m2:c1", "The user likes coffee.", 1, ["coffee"]),
    claim("m3:c1", "The user moved to Bangalore.", 2, ["Bangalore"], "residence"),
    claim("m4:c1", "Leah actually lives in Valencia now.", 3, ["Leah", "Valencia"]),
    claim("m5:c1", "The user likes coffee.", 4, ["coffee"]),
    claim("m6:c1", "The museum is closed Monday.", 5, ["museum"]),
    claim("m7:c1", "The museum is open Monday.", 6, ["museum"]),
]
RELATIONS = [rel("m4:c1", "m1:c2", "UPDATES"), rel("m7:c1", "m6:c1", "CONTRADICTS")]


def _snapshot(c: Consolidator):
    return ([m.to_dict() for m in c.memories.values()], [s.to_dict() for s in c.sources],
            [l.to_dict() for l in c.links], [d.to_dict() for d in c.decisions])


def test_deterministic_regardless_of_input_order():
    expected = _snapshot(consolidate(CONV, STREAM, RELATIONS))
    for seed in range(5):
        claims, relations = STREAM[:], RELATIONS[:]
        random.Random(seed).shuffle(claims)
        random.Random(seed).shuffle(relations)
        assert _snapshot(consolidate(CONV, claims, relations)) == expected


def test_incremental_application_matches_a_single_batch():
    whole = consolidate(CONV, STREAM, RELATIONS)
    stepwise = Consolidator(CONV)
    stepwise.apply(STREAM[:3], [])
    stepwise.apply(STREAM[3:], RELATIONS)
    assert _snapshot(stepwise) == _snapshot(whole)


def test_mixed_stream_outcome_and_no_memory_is_ever_deleted():
    c = consolidate(CONV, STREAM, RELATIONS)
    assert actions(c) == [CREATE, CREATE, CREATE, SUPERSEDE, UPDATE, KEEP, CREATE, CONTRADICTION]
    assert len(c.memories) == len(STREAM) - 1  # only the restatement shares a memory
    claimed = {s.claim_id for s in c.sources}
    assert claimed == {x.claim_id for x in STREAM}  # every claim stays traceable
    assert [m.content for m in c.current()] == [
        "The user likes coffee.", "The user moved to Bangalore.", "Leah actually lives in Valencia now.",
        "The museum is closed Monday.", "The museum is open Monday."]
    result = c.apply([])
    assert result.decisions == [] and len(c.memories) == 7


def test_result_summaries():
    c = Consolidator(CONV)
    result = c.apply(STREAM, RELATIONS)
    assert result.created == ["mem:m1:c1", "mem:m1:c2", "mem:m2:c1", "mem:m3:c1", "mem:m4:c1", "mem:m6:c1",
                              "mem:m7:c1"]
    assert result.kept == [("m5:c1", "mem:m2:c1")]
    assert result.superseded == ["mem:m1:c1", "mem:m1:c2"]
    assert result.contradictions == [("mem:m7:c1", "mem:m6:c1")]
    assert result.unresolved == [] and result.ignored == []
    assert [l.type for l in result.links] == ["SUPERSEDES", "UPDATES", "CONTRADICTS"]
