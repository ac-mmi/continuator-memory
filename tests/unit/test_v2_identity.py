import random

import numpy as np

from continuator_memory.models import parse_conversation
from continuator_memory.v2.consolidation import Consolidator, consolidate
from continuator_memory.v2.identity import IdentityRegistry, claim_attribute
from continuator_memory.v2.memory import (
    CONTRADICTION,
    CREATE,
    HISTORICAL,
    IDENTITY_AMBIGUOUS,
    RELATION_IDENTITY_CONFLICT,
    RELATION_TARGET_AMBIGUOUS,
    RELATION_TARGET_MISSING,
    SLOT_AMBIGUOUS,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_SUPERSEDED,
    SUBJECT_MISSING,
    SUPERSEDE,
    UPDATE,
    Decision,
)
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.models import SUPPORT_IDENTITY, Claim, Provenance, Relation, Temporal

CONV = "conv"


def claim(cid, content, position, entities=(), claim_type=None, metadata=None, temporal=None, polarity="positive",
          support="SOURCE", resolved_from=()):
    mid = cid.rsplit(":c", 1)[0]
    return Claim(cid, content, Provenance(CONV, mid, position, "user"), tuple(entities), (),
                 Temporal(temporal) if temporal else None, polarity, claim_type, support, tuple(resolved_from),
                 metadata or {})


def rel(src, dst, rtype, n=1):
    mid = src.rsplit(":c", 1)[0]
    return Relation(f"{mid}:r{n}", src, dst, rtype, mid, "evidence")


def table(c: Consolidator):
    return [(m.content, m.status) for m in c._ordered()]


def links(c: Consolidator):
    return [(l.newer_memory_id, l.older_memory_id, l.type) for l in c.links]


def actions(c: Consolidator):
    return [d.action for d in c.decisions]


def codes(c: Consolidator, i: int):
    return c.decisions[i].unresolved_codes


def sources(c: Consolidator):
    return {m.memory_id: [s.claim_id for s in c.sources_of(m.memory_id)] for m in c._ordered()}


ALIAS = {"aliases": {"Sam": "Samuel"}}
TEAM_LEAD = {"subject": "the user", "attribute": "team lead"}


# ---- identity registry -------------------------------------------------------------


def test_exact_name_is_the_identity_and_nothing_is_fuzzy():
    r = IdentityRegistry()
    c = claim("a:c1", "x", 0)
    assert r.resolve("Sam", c) == ("sam", None)
    assert r.resolve("SAM", c) == ("sam", None)
    assert r.resolve("Samuel", c) == ("samuel", None)
    assert r.resolve("Mom", c) == ("mom", None)
    assert r.resolve("I", c) == ("user", None)


def test_explicit_alias_applies_from_its_declaration_onwards():
    r = IdentityRegistry()
    before = claim("a:c1", "Sam has a cat.", 0, ["Sam"])
    assert r.identify(before).subject == "sam"
    r.identify(claim("a:c2", "Samuel, who the user calls Sam, works in finance.", 1, ["Samuel", "Sam"], metadata=ALIAS))
    assert r.identify(claim("a:c3", "Sam moved to Pune.", 2, ["Sam", "Pune"])).subject == "samuel"


def test_same_nickname_for_different_entities_and_ambiguous_alias():
    r = IdentityRegistry()
    a = r.identify(claim("a:c1", "Sam has a cat.", 0, ["Sam"], metadata={"entity_ids": {"Sam": "p1"}}))
    b = r.identify(claim("a:c2", "Sam has a dog.", 1, ["Sam"], metadata={"entity_ids": {"Sam": "p2"}}))
    assert (a.subject, b.subject) == ("id:p1", "id:p2")
    later = r.identify(claim("a:c3", "Sam moved to Pune.", 2, ["Sam"]))
    assert later.subject is None and later.problems[0].startswith(IDENTITY_AMBIGUOUS)

    r = IdentityRegistry()
    r.identify(claim("b:c1", "x", 0, metadata={"aliases": {"Sam": "Samuel"}}))
    r.identify(claim("b:c2", "x", 1, metadata={"aliases": {"Sam": "Samantha"}}))
    ambiguous = r.identify(claim("b:c3", "Sam moved to Pune.", 2, ["Sam"]))
    assert ambiguous.subject is None and "declared an alias of ['samantha', 'samuel']" in ambiguous.problems[0]


def test_subject_precedence_and_structural_attribute_normalization():
    r = IdentityRegistry()
    by_id = r.identify(claim("a:c1", "Samuel moved.", 0, ["Samuel"], metadata={"subject_id": "p9", "subject": "Alex"}))
    assert (by_id.subject, by_id.explicit_subject) == ("id:p9", "id:p9")
    by_meta = r.identify(claim("a:c2", "Hana leads.", 1, ["Hana"], metadata={"subject": "The User"}))
    assert (by_meta.subject, by_meta.explicit_subject) == ("user", "user")
    positional = r.identify(claim("a:c3", "Hana leads.", 2, ["Hana"]))
    assert (positional.subject, positional.explicit_subject) == ("hana", None)
    assert claim_attribute(claim("a:c4", "x", 0, claim_type="x", metadata={"attribute": "  Team-Lead "})) == "team lead"
    assert claim_attribute(claim("a:c5", "x", 0, claim_type="Team Lead")) == "team lead"
    assert claim_attribute(claim("a:c6", "x", 0, claim_type="Team Lead"), claim_type_slots=False) is None


def test_malformed_metadata_is_ignored():
    r = IdentityRegistry()
    ident = r.identify(claim("a:c1", "Sam has a cat.", 0, ["Sam"],
                             metadata={"aliases": ["Sam"], "entity_ids": {"Sam": 3}, "subject": "  "}))
    assert ident.subject == "sam" and ident.problems == ()


# ---- negative cases: must not merge ----------------------------------------------------


def test_N1_N2_similar_names_without_alias_stay_separate():
    c = consolidate(CONV, [claim("m1:c1", "Sam has a cat.", 0, ["Sam", "cat"], "pet"),
                           claim("m2:c1", "Samuel now has a dog.", 1, ["Samuel", "dog"], "pet"),
                           claim("m3:c1", "Sam works at Google.", 2, ["Sam", "Google"], "location"),
                           claim("m4:c1", "Samuel now lives in Delhi.", 3, ["Samuel", "Delhi"], "location")])
    assert actions(c) == [CREATE] * 4
    assert {s for _, s in table(c)} == {STATUS_CURRENT} and links(c) == []
    assert [m.subject for m in c._ordered()] == ["sam", "samuel", "sam", "samuel"]


def test_N3_different_people_with_the_same_attribute():
    c = consolidate(CONV, [claim("m1:c1", "Alex is the team lead.", 0, ["Alex"], "team lead"),
                           claim("m2:c1", "Sam is now the team lead.", 1, ["Sam"], "team lead")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert {s for _, s in table(c)} == {STATUS_CURRENT}


def test_N4_same_subject_different_explicit_attributes():
    c = consolidate(CONV, [
        claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"], "preference", TEAM_LEAD),
        claim("m2:c1", "The user's preferred meeting time is now Thursday.", 1, ["Thursday"], "preference",
              {"subject": "the user", "attribute": "preferred meeting time"})])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert [m.attribute for m in c._ordered()] == ["team lead", "preferred meeting time"]


def test_N5_different_relatives_with_the_same_city():
    c = consolidate(CONV, [claim("m1:c1", "The user's brother lives in Delhi.", 0, ["Delhi"], "residence"),
                           claim("m2:c1", "The user's friend lives in Delhi now.", 1, ["Delhi"], "residence")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert {s for _, s in table(c)} == {STATUS_CURRENT}
    assert codes(c, 1) == (SUBJECT_MISSING,)


def test_N6_same_pet_name_for_different_owners():
    c = consolidate(CONV, [claim("m1:c1", "Sam's cat is Biscuit.", 0, ["Sam", "Biscuit"], "pet"),
                           claim("m2:c1", "Alex's cat is now Biscuit.", 1, ["Alex", "Biscuit"], "pet")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert len(c.current()) == 2


def test_N7_unresolved_pronoun_is_identity_ambiguous():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah", "Madrid"], "residence"),
                           claim("m2:c1", "She moved to Bangalore.", 1, ["Bangalore"], "residence")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert codes(c, 1) == (IDENTITY_AMBIGUOUS,)


def test_N8_change_without_subject_never_picks_one_of_several_slots():
    c = consolidate(CONV, [
        claim("m1:c1", "Sam's team lead is Marco.", 0, ["Sam", "Marco"], metadata={"subject": "Sam", "attribute": "team lead"}),
        claim("m2:c1", "Alex's team lead is Priya.", 1, ["Alex", "Priya"], metadata={"subject": "Alex", "attribute": "team lead"}),
        claim("m3:c1", "The team lead is changing.", 2, ["team lead"]),
        claim("m4:c1", "The team lead changed to Dana.", 3, ["Dana"], "team lead")])
    assert actions(c) == [CREATE] * 4 and links(c) == []
    assert table(c) == [("Sam's team lead is Marco.", STATUS_CURRENT), ("Alex's team lead is Priya.", STATUS_CURRENT),
                        ("The team lead is changing.", STATUS_CURRENT), ("The team lead changed to Dana.", STATUS_CURRENT)]
    assert codes(c, 3) == (SUBJECT_MISSING,) and "2 memories left unchanged" in c.decisions[3].unresolved


def test_N9_relation_to_a_nonexistent_claim():
    c = consolidate(CONV, [claim("m1:c1", "Sam lives in Delhi.", 0, ["Sam"], "residence"),
                           claim("m2:c1", "Sam lives in Pune now.", 1, ["Sam"])],
                    [rel("m2:c1", "m0:c9", "UPDATES")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert codes(c, 1) == (RELATION_TARGET_MISSING,)


def test_N10_relation_with_two_equally_plausible_targets():
    c = consolidate(CONV, [claim("m1:c1", "The user speaks Spanish.", 0, claim_type="language"),
                           claim("m2:c1", "The user speaks French.", 1, claim_type="language"),
                           claim("m3:c1", "The user now speaks German.", 2, claim_type="language")],
                    [rel("m3:c1", "m0:c9", "UPDATES")])
    assert actions(c) == [CREATE, CREATE, CREATE] and links(c) == []
    assert {s for _, s in table(c)} == {STATUS_CURRENT}
    assert codes(c, 2) == (RELATION_TARGET_AMBIGUOUS,)


def test_slot_ambiguity_is_reported_with_its_code():
    c = consolidate(CONV, [claim("m1:c1", "The user speaks Spanish.", 0, claim_type="language"),
                           claim("m2:c1", "The user speaks French.", 1, claim_type="language"),
                           claim("m3:c1", "The user now speaks German.", 2, claim_type="language")])
    assert codes(c, 2) == (SLOT_AMBIGUOUS,)


# ---- slots: explicit attributes vs claim_type ------------------------------------------


def test_generic_claim_type_does_not_merge_when_explicit_attributes_differ():
    c = consolidate(CONV, [
        claim("m1:c1", "The user likes tea.", 0, claim_type="preference", metadata={"attribute": "drink"}),
        claim("m2:c1", "The user now prefers a window seat.", 1, claim_type="preference",
              metadata={"attribute": "seat"})])
    assert actions(c) == [CREATE, CREATE] and links(c) == []


def test_strict_mode_ignores_claim_type_as_a_slot():
    claims = [claim("m1:c1", "The user likes tea.", 0, claim_type="preference"),
              claim("m2:c1", "The user now prefers a window seat.", 1, claim_type="preference")]
    strict = consolidate(CONV, claims, claim_type_slots=False)
    assert actions(strict) == [CREATE, CREATE] and links(strict) == []
    # Phase 2A default, kept for compatibility: equal claim_types still form one slot.
    default = consolidate(CONV, claims)
    assert actions(default) == [CREATE, SUPERSEDE]


def test_strict_mode_still_versions_explicit_attributes():
    c = consolidate(CONV, [claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"], metadata=TEAM_LEAD),
                           claim("m2:c1", "Hana is taking over as the user's team lead.", 1, ["Hana"],
                                 metadata=TEAM_LEAD)], claim_type_slots=False)
    assert actions(c) == [CREATE, SUPERSEDE]


def test_no_semantic_attribute_equivalence():
    c = consolidate(CONV, [
        claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"], metadata=TEAM_LEAD),
        claim("m2:c1", "The user's manager is now Hana.", 1, ["Hana"], metadata={"subject": "the user", "attribute": "manager"})])
    assert actions(c) == [CREATE, CREATE] and links(c) == []


# ---- positive cases -------------------------------------------------------------------


def test_P1_exact_same_entity_and_slot():
    c = consolidate(CONV, [claim("m1:c1", "Sam lives in Delhi.", 0, ["Sam", "Delhi"], "residence"),
                           claim("m2:c1", "Sam lives in Pune now.", 1, ["Sam", "Pune"], "residence")])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert table(c) == [("Sam lives in Delhi.", STATUS_SUPERSEDED), ("Sam lives in Pune now.", STATUS_CURRENT)]


def test_P2_section8_team_lead_three_cases():
    marco = claim("m1:c1", "The user's team lead is Marco.", 0, ["Marco"])
    hana = claim("m2:c1", "Hana is taking over as the user's team lead.", 1, ["Hana"])
    with_relation = consolidate(CONV, [marco, hana], [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(with_relation) == [CREATE, UPDATE]
    assert links(with_relation) == [("mem:m2:c1", "mem:m1:c1", "UPDATES")]

    marco_m = claim("m1:c1", marco.content, 0, ["Marco"], metadata=TEAM_LEAD)
    hana_m = claim("m2:c1", hana.content, 1, ["Hana"], metadata=TEAM_LEAD)
    with_metadata = consolidate(CONV, [marco_m, hana_m])
    assert actions(with_metadata) == [CREATE, SUPERSEDE]
    assert table(with_metadata) == [(marco.content, STATUS_SUPERSEDED), (hana.content, STATUS_CURRENT)]

    neither = consolidate(CONV, [marco, hana])
    assert actions(neither) == [CREATE, CREATE] and links(neither) == []


def test_P3_cross_message_alias_only_when_declared():
    samuel = claim("m1:c1", "Samuel lives in Delhi.", 0, ["Samuel", "Delhi"], "residence")
    calls = claim("m2:c1", "The user calls Samuel Sam.", 1, ["Samuel", "Sam"], metadata=ALIAS)
    sam = claim("m3:c1", "Sam moved to Pune.", 2, ["Sam", "Pune"], "residence")
    without = consolidate(CONV, [samuel, sam])
    assert actions(without) == [CREATE, CREATE] and links(without) == []
    assert [m.subject for m in without._ordered()] == ["samuel", "sam"]
    with_alias = consolidate(CONV, [samuel, calls, sam])
    assert actions(with_alias) == [CREATE, CREATE, SUPERSEDE]
    assert with_alias.memories["mem:m3:c1"].subject == "samuel"
    assert with_alias.memories["mem:m3:c1"].entities == ("samuel", "pune")
    assert links(with_alias) == [("mem:m3:c1", "mem:m1:c1", "SUPERSEDES")]


def test_P3_work_with_samuel_then_sam_takes_over_stays_separate_without_alias():
    c = consolidate(CONV, [claim("m1:c1", "The user works with Samuel.", 0, ["Samuel"]),
                           claim("m2:c1", "Sam is taking over the project.", 1, ["Sam", "project"])])
    assert [m.entities for m in c._ordered()] == [("samuel",), ("sam", "project")]
    assert links(c) == []


def test_P3_alias_declared_after_a_claim_is_not_retroactive():
    c = consolidate(CONV, [claim("m1:c1", "Sam lives in Delhi.", 0, ["Sam"], "residence"),
                           claim("m2:c1", "The user calls Samuel Sam.", 1, ["Samuel", "Sam"], metadata=ALIAS),
                           claim("m3:c1", "Samuel moved to Pune.", 2, ["Samuel"], "residence")])
    assert actions(c) == [CREATE, CREATE, CREATE]
    assert [m.subject for m in c._ordered()] == ["sam", "user", "samuel"]


def test_pronoun_already_resolved_by_phase1_is_a_safe_identity():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah", "Madrid"], "residence"),
                           claim("m2:c1", "Leah moved to Bangalore.", 1, ["Leah", "Bangalore"], "residence",
                                 support=SUPPORT_IDENTITY, resolved_from=("m1",))])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert table(c) == [("Leah lives in Madrid.", STATUS_SUPERSEDED), ("Leah moved to Bangalore.", STATUS_CURRENT)]


def test_P8_explicit_entity_id_carried_through_claims():
    c = consolidate(CONV, [
        claim("m1:c1", "Sam lives in Delhi.", 0, ["Sam"], "residence", {"entity_ids": {"Sam": "p7"}}),
        claim("m2:c1", "Sam moved to Pune.", 1, ["Sam"], "residence"),
        claim("m3:c1", "Samuel moved to Goa.", 2, ["Samuel"], "residence", {"subject_id": "p7"})])
    assert actions(c) == [CREATE, SUPERSEDE, SUPERSEDE]
    assert [m.subject for m in c._ordered()] == ["id:p7"] * 3
    assert [m.content for m in c.current()] == ["Samuel moved to Goa."]


# ---- relation targets --------------------------------------------------------------------


def test_relation_targets_by_exact_claim_id():
    c = consolidate(CONV, [claim("m1:c1", "The museum is closed Monday.", 0, ["museum"]),
                           claim("m2:c1", "The museum is open Monday.", 1, ["museum"]),
                           claim("m3:c1", "The museum's Monday hours were posted online.", 2, ["museum"]),
                           claim("m4:c1", "Leah lives in Madrid.", 3, ["Leah"]),
                           claim("m5:c1", "Leah lives in Valencia now.", 4, ["Leah"])],
                    [rel("m2:c1", "m1:c1", "CONTRADICTS"), rel("m3:c1", "m2:c1", "REFERS_TO"),
                     rel("m5:c1", "m4:c1", "UPDATES")])
    assert actions(c) == [CREATE, CONTRADICTION, CREATE, CREATE, UPDATE]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "CONTRADICTS"), ("mem:m3:c1", "mem:m2:c1", "REFERS_TO"),
                        ("mem:m5:c1", "mem:m4:c1", "UPDATES")]
    assert [s for _, s in table(c)][:2] == [STATUS_CONTESTED, STATUS_CONTESTED]


def test_relation_target_by_exact_memory_id():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah"]),
                           claim("m2:c1", "Leah lives in Valencia now.", 1, ["Leah"])],
                    [rel("m2:c1", "mem:m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "UPDATES")]


def test_relation_target_by_identity_when_exactly_one_is_compatible():
    c = consolidate(CONV, [claim("m1:c1", "Sam lives in Delhi.", 0, ["Sam"], "residence"),
                           claim("m2:c1", "Alex lives in Delhi.", 1, ["Alex"], "residence"),
                           claim("m3:c1", "Sam lives in Pune now.", 2, ["Sam"], "residence")],
                    [rel("m3:c1", "m0:c9", "UPDATES")])
    assert actions(c) == [CREATE, CREATE, UPDATE]
    assert links(c) == [("mem:m3:c1", "mem:m1:c1", "UPDATES")]
    assert table(c)[1] == ("Alex lives in Delhi.", STATUS_CURRENT)


def test_refers_to_by_identity_when_exactly_one_is_compatible():
    c = consolidate(CONV, [claim("m1:c1", "Sam's cat is Biscuit.", 0, ["Sam", "Biscuit"]),
                           claim("m2:c1", "Biscuit is asleep on the sofa.", 1, ["Biscuit"])],
                    [rel("m2:c1", "m0:c9", "REFERS_TO")])
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "REFERS_TO")]
    assert c.decisions[1].unresolved is None


def test_ambiguous_contradicts_target_preserves_both():
    c = consolidate(CONV, [claim("m1:c1", "The user speaks Spanish.", 0, claim_type="language"),
                           claim("m2:c1", "The user speaks French.", 1, claim_type="language"),
                           claim("m3:c1", "The user does not speak French.", 2, claim_type="language",
                                 polarity="negative")],
                    [rel("m3:c1", "m0:c9", "CONTRADICTS")])
    assert actions(c) == [CREATE, CREATE, CREATE] and links(c) == []
    assert {s for _, s in table(c)} == {STATUS_CURRENT}
    assert codes(c, 2) == (RELATION_TARGET_AMBIGUOUS,)


def test_conflicting_explicit_identities_block_updates():
    c = consolidate(CONV, [claim("m1:c1", "Sam has a cat.", 0, ["Sam"], metadata={"subject": "Sam"}),
                           claim("m2:c1", "Alex now has a cat.", 1, ["Alex"], metadata={"subject": "Alex"})],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert table(c) == [("Sam has a cat.", STATUS_CURRENT), ("Alex now has a cat.", STATUS_CURRENT)]
    assert codes(c, 1) == (RELATION_IDENTITY_CONFLICT,)


def test_conflicting_explicit_ids_block_contradicts():
    c = consolidate(CONV, [claim("m1:c1", "Sam is vegetarian.", 0, ["Sam"], metadata={"subject_id": "p1"}),
                           claim("m2:c1", "Sam is not vegetarian.", 1, ["Sam"], metadata={"subject_id": "p2"},
                                 polarity="negative")],
                    [rel("m2:c1", "m1:c1", "CONTRADICTS")])
    assert actions(c) == [CREATE, CREATE] and links(c) == []
    assert codes(c, 1) == (RELATION_IDENTITY_CONFLICT,)


def test_explicit_alias_reconciles_explicit_subjects_for_updates():
    c = consolidate(CONV, [claim("m1:c1", "Samuel lives in Delhi.", 0, ["Samuel"], metadata={"subject": "Samuel"}),
                           claim("m2:c1", "Sam lives in Pune now.", 1, ["Sam"], metadata={"subject": "Sam", **ALIAS})],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE]


def test_unsafe_refers_to_targets_are_never_guessed():
    c = consolidate(CONV, [
        claim("m1:c1", "Sam has a cat.", 0, ["Sam", "cat"]),
        claim("m2:c1", "Sam works at Google.", 1, ["Sam", "Google"]),
        claim("m3:c1", "Alex has a cat.", 2, ["Alex", "cat"]),
        claim("m4:c1", "Priya is a team lead.", 3, ["Priya", "team lead"]),
        claim("m5:c1", "Marco is a team lead.", 4, ["Marco", "team lead"]),
        claim("m6:c1", "Sam mentioned this yesterday.", 5, ["Sam"]),  # two Sam memories
        claim("m7:c1", "The cat is grey.", 6, ["cat"]),  # two cats
        claim("m8:c1", "The team lead approved the plan.", 7, ["team lead"]),  # two team leads
        claim("m9:c1", "She said it again.", 8)],  # unresolved pronouns
        [rel("m6:c1", "m0:c1", "REFERS_TO"), rel("m7:c1", "m0:c2", "REFERS_TO"),
         rel("m8:c1", "m0:c3", "REFERS_TO"), rel("m9:c1", "m0:c4", "REFERS_TO")])
    assert [l for l in links(c) if l[2] == "REFERS_TO"] == []
    assert [codes(c, i) for i in (5, 6, 7, 8)] == [(RELATION_TARGET_AMBIGUOUS,), (RELATION_TARGET_AMBIGUOUS,),
                                                   (RELATION_TARGET_AMBIGUOUS,), (RELATION_TARGET_MISSING,)]
    assert len(c.memories) == 9


def test_refers_to_with_explicit_target_resolves_a_pronoun_claim():
    c = consolidate(CONV, [claim("m1:c1", "Leah adopted a cat named Biscuit.", 0, ["Leah", "Biscuit"]),
                           claim("m2:c1", "She loves it.", 1)],
                    [rel("m2:c1", "m1:c1", "REFERS_TO")])
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "REFERS_TO")]
    assert table(c) == [("Leah adopted a cat named Biscuit.", STATUS_CURRENT), ("She loves it.", STATUS_CURRENT)]


# ---- temporal interaction (phase 2B decides currentness) ----------------------------------


def test_identity_match_defers_ordering_to_temporal_semantics():
    samuel = claim("m1:c1", "Samuel moved to Bangalore in 2024.", 0, ["Samuel"], "residence", temporal="in 2024")
    sam_earlier = claim("m2:c1", "Sam moved to Mumbai in 2023.", 1, ["Sam"], "residence", ALIAS, temporal="in 2023")
    sam_future = claim("m3:c1", "Sam will move to Goa next year.", 2, ["Sam"], "residence", temporal="next year")
    c = consolidate(CONV, [samuel, sam_earlier, sam_future])
    assert actions(c) == [CREATE, HISTORICAL, CREATE]
    assert table(c) == [("Samuel moved to Bangalore in 2024.", STATUS_CURRENT),
                        ("Sam moved to Mumbai in 2023.", STATUS_SUPERSEDED),
                        ("Sam will move to Goa next year.", STATUS_FUTURE)]
    assert links(c) == [("mem:m1:c1", "mem:m2:c1", "SUPERSEDES")]
    assert [m.subject for m in c._ordered()] == ["samuel"] * 3


# ---- incremental vs full replay -------------------------------------------------------------


STREAM = [
    claim("m1:c1", "Samuel lives in Delhi.", 0, ["Samuel"], "residence"),
    claim("m2:c1", "The user calls Samuel Sam.", 1, ["Samuel", "Sam"], metadata=ALIAS),
    claim("m3:c1", "Sam moved to Pune.", 2, ["Sam"], "residence"),
    claim("m4:c1", "Leah lives in Madrid.", 3, ["Leah"]),
    claim("m5:c1", "Leah lives in Valencia now.", 4, ["Leah"]),
    claim("m6:c1", "The museum is closed Monday.", 5, ["museum"]),
    claim("m7:c1", "The museum is open Monday.", 6, ["museum"]),
    claim("m8:c1", "Sam lives in Pune.", 7, ["Sam"], "residence"),
]
RELATIONS = [rel("m5:c1", "m4:c1", "UPDATES"), rel("m7:c1", "m6:c1", "CONTRADICTS")]


def _snapshot(c: Consolidator):
    return ([m.to_dict() for m in c.memories.values()], [s.to_dict() for s in c.sources],
            [l.to_dict() for l in c.links], [d.to_dict() for d in c.decisions])


def test_P9_incremental_batches_equal_full_replay():
    full = _snapshot(consolidate(CONV, STREAM, RELATIONS))
    stepwise = Consolidator(CONV)
    stepwise.add(STREAM[:2])
    stepwise.add(STREAM[2:5], RELATIONS[:1])
    stepwise.add(STREAM[5:], RELATIONS[1:])
    assert _snapshot(stepwise) == full


def test_incremental_out_of_order_batch_replays_to_the_same_result():
    full = _snapshot(consolidate(CONV, STREAM, RELATIONS))
    for seed in range(5):
        claims = STREAM[:]
        random.Random(seed).shuffle(claims)
        stepwise = Consolidator(CONV)
        for i in range(0, len(claims), 3):
            stepwise.add(claims[i:i + 3], RELATIONS)
        assert _snapshot(stepwise) == full


def test_incremental_late_relation_is_applied_like_a_full_replay():
    full = consolidate(CONV, STREAM, RELATIONS)
    stepwise = Consolidator(CONV)
    stepwise.add(STREAM)
    result = stepwise.add([], RELATIONS)
    assert _snapshot(stepwise) == _snapshot(full)
    assert result.decisions == [] and result.unresolved_relations == []
    assert stepwise.memories["mem:m4:c1"].status == STATUS_SUPERSEDED


def test_incremental_add_ignores_resent_claims():
    stepwise = Consolidator(CONV)
    stepwise.add(STREAM, RELATIONS)
    result = stepwise.add(STREAM, RELATIONS)
    assert result.decisions == []
    assert _snapshot(stepwise) == _snapshot(consolidate(CONV, STREAM, RELATIONS))


# ---- audit and persistence --------------------------------------------------------------------


def test_unresolved_codes_are_stable_prefixes():
    d = Decision("x:c1", CREATE, "mem:x:c1",
                 unresolved=f"{SUBJECT_MISSING}: a; {RELATION_TARGET_MISSING}: b; free text; {SUBJECT_MISSING}: c")
    assert d.unresolved_codes == (SUBJECT_MISSING, RELATION_TARGET_MISSING)
    assert Decision("x:c2", CREATE, None).unresolved_codes == ()


def test_identity_fields_round_trip_through_the_store(store):
    messages = [{"id": x.provenance.message_id, "role": "user", "content": x.content} for x in STREAM]
    store.save_conversation(parse_conversation({"conversation_id": CONV, "messages": messages}), [],
                            np.zeros((0, 0), dtype=np.float32), "none")
    c = consolidate(CONV, STREAM, RELATIONS)
    ms = MemoryStore(store.connection)
    ms.save(c)
    saved = ms.get_memories(CONV)
    assert [m.to_dict() for m in saved] == [m.to_dict() for m in c._ordered()]
    assert {(m.subject, m.attribute) for m in saved if m.claim_id == "m3:c1"} == {("samuel", "residence")}
    assert [d.unresolved_codes for d in ms.get_decisions(CONV)] == [d.unresolved_codes for d in c.decisions]
