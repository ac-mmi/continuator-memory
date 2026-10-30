import random

import numpy as np
import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.v2.consolidation import Consolidator, consolidate
from continuator_memory.v2.memory import (
    CONTRADICTION,
    CREATE,
    HISTORICAL,
    KEEP,
    STATUS_CONDITIONAL,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    SUPERSEDE,
    UPDATE,
)
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.models import Claim, Provenance, Relation, Temporal
from continuator_memory.v2.temporal import (
    TIME_AMBIGUOUS_PAST,
    TIME_CONDITIONAL,
    TIME_FUTURE,
    TIME_MIXED,
    TIME_PAST,
    TIME_PRESENT,
    TIME_TRANSITION,
    TIME_UNSPECIFIED,
    before,
    classify,
    temporal_bounds,
)

CONV = "conv"


def claim(cid, content, position, entities=(), claim_type=None, polarity="positive", temporal=None,
          resolved=None, qualifiers=(), metadata=None, ts=None):
    mid = cid.rsplit(":c", 1)[0]
    return Claim(cid, content, Provenance(CONV, mid, position, "user", ts), tuple(entities), tuple(qualifiers),
                 Temporal(temporal, resolved) if temporal else None, polarity, claim_type, metadata=metadata or {})


def rel(src, dst, rtype, n=1):
    mid = src.rsplit(":c", 1)[0]
    return Relation(f"{mid}:r{n}", src, dst, rtype, mid, "evidence")


def table(c: Consolidator):
    """(content, status, valid_from, valid_until) per memory, in conversation order."""
    return [(m.content, m.status, m.valid_from, m.valid_until) for m in c._ordered()]


def links(c: Consolidator):
    return [(l.newer_memory_id, l.older_memory_id, l.type) for l in c.links]


def sources(c: Consolidator):
    return {m.memory_id: [s.claim_id for s in c.sources_of(m.memory_id)] for m in c._ordered()}


def current(c: Consolidator):
    return [m.content for m in c.current()]


def actions(c: Consolidator):
    return [d.action for d in c.decisions]


RES = "residence"
DELHI = claim("m1:c1", "The user lives in Delhi.", 0, ["Delhi"], RES)


# ---- classification and dates ----------------------------------------------------


@pytest.mark.parametrize("content, temporal, expected", [
    ("The user used to live in Delhi.", None, TIME_PAST),
    ("The user previously worked at Google.", None, TIME_PAST),
    ("Formerly, the user's team lead was Marco.", None, TIME_PAST),
    ("Last year the user preferred tea.", "Last year", TIME_PAST),
    ("The user lived in Delhi earlier.", None, TIME_PAST),
    ("Before, the user worked at Google.", None, TIME_PAST),
    ("The user lived in Delhi before.", None, TIME_PAST),
    ("The user lived in Delhi.", None, TIME_PAST),
    ("The user no longer lives in Delhi.", None, TIME_TRANSITION),
    ("The user moved to Bangalore in 2024.", "in 2024", TIME_TRANSITION),
    ("The user moved to Bangalore in May 2024.", "in May 2024", TIME_TRANSITION),  # "May" is a month
    ("The user lives in Bangalore now.", None, TIME_PRESENT),
    ("The user's current team lead is Hana.", None, TIME_PRESENT),
    ("The user has lived in Pune since 2021.", "since 2021", TIME_PRESENT),
    ("The user will move to Bangalore next month.", "next month", TIME_FUTURE),
    ("The user is going to leave the company in June.", "in June", TIME_FUTURE),
    ("Tomorrow the user will meet Sam.", "Tomorrow", TIME_FUTURE),
    ("If the offer works out, the user will move to Bangalore.", None, TIME_CONDITIONAL),
    ("The user might move next month.", "next month", TIME_CONDITIONAL),
    ("The user may switch teams.", None, TIME_CONDITIONAL),
    ("The user used to live in Delhi, but now the user lives in Bangalore.", None, TIME_MIXED),
    ("The user was in Delhi.", None, TIME_AMBIGUOUS_PAST),
    ("The user likes coffee.", None, TIME_UNSPECIFIED),
])
def test_classify_temporal_meaning(content, temporal, expected):
    assert classify(claim("x:c1", content, 0, temporal=temporal)) == expected


@pytest.mark.parametrize("expression, resolved, transition, expected", [
    ("in 2024", None, True, ("2024", None, "2024")),
    ("in 2024", None, False, (None, None, "2024")),
    ("on March 3", "2026-03-03", True, ("2026-03-03", None, "2026-03-03")),
    ("in May 2024", None, True, ("2024-05", None, "2024-05")),
    ("on March 3, 2024", None, False, (None, None, "2024-03-03")),
    ("from 2020 to 2023", None, False, ("2020", "2023", None)),
    ("between 2019 and 2021", None, False, ("2019", "2021", None)),
    ("2020-2023", None, False, ("2020", "2023", None)),
    ("since 2021", None, False, ("2021", None, None)),
    ("starting March 2025", None, False, ("2025-03", None, None)),
    ("until 2023", None, False, (None, "2023", None)),
    ("last year", None, False, (None, None, None)),  # relative: only Phase 1 may resolve it
    ("in March 2024", "2024-03-01", True, ("2024-03", None, "2024-03")),  # stated precision wins
])
def test_temporal_bounds_keep_stated_precision(expression, resolved, transition, expected):
    b = temporal_bounds(claim("x:c1", "x", 0, temporal=expression, resolved=resolved), transition)
    assert (b.valid_from, b.valid_until, b.event_time) == expected


def test_dates_are_ordered_only_when_their_intervals_do_not_overlap():
    assert before("2023", "2024") and before("2024-02", "2024-03-01")
    assert not before("2024", "2024-03-03") and not before("2024-03-03", "2024")
    assert not before("2024", "2024")


# ---- A. past-state claims ------------------------------------------------------


@pytest.mark.parametrize("content, temporal", [
    ("The user used to live in Delhi.", None),
    ("The user previously worked at Google.", None),
    ("Formerly, the user's team lead was Marco.", None),
    ("Last year the user preferred tea.", "Last year"),
    ("The user lived in Delhi earlier.", None),
    ("Before, the user worked at Google.", None),
])
def test_A_past_state_is_historical_even_when_newest(content, temporal):
    bangalore = claim("m1:c1", "The user lives in Bangalore.", 0, claim_type=RES)
    c = consolidate(CONV, [bangalore, claim("m2:c1", content, 1, claim_type=RES, temporal=temporal)])
    assert actions(c) == [CREATE, HISTORICAL]
    assert table(c) == [("The user lives in Bangalore.", STATUS_CURRENT, None, None),
                        (content, STATUS_HISTORICAL, None, None)]
    assert current(c) == ["The user lives in Bangalore."]
    assert links(c) == []
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_A_lived_then_moved_keeps_delhi_historical_and_bangalore_current():
    c = consolidate(CONV, [claim("m1:c1", "The user lived in Delhi.", 0, ["Delhi"], RES),
                           claim("m2:c1", "The user moved to Bangalore in 2024.", 1, ["Bangalore"], RES,
                                 temporal="in 2024")])
    assert actions(c) == [HISTORICAL, CREATE]
    assert table(c) == [("The user lived in Delhi.", STATUS_HISTORICAL, None, None),
                        ("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None)]
    assert current(c) == ["The user moved to Bangalore in 2024."]
    assert links(c) == [] and sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_A_no_longer_makes_the_old_state_non_current():
    c = consolidate(CONV, [DELHI, claim("m2:c1", "The user no longer lives in Delhi.", 1, ["Delhi"], RES,
                                        polarity="negative")])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, None),
                        ("The user no longer lives in Delhi.", STATUS_CURRENT, None, None)]
    assert current(c) == ["The user no longer lives in Delhi."]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]
    assert sources(c)["mem:m1:c1"] == ["m1:c1"]


def test_A_used_to_but_now_split_into_two_claims_is_not_collapsed():
    c = consolidate(CONV, [DELHI,
                           claim("m2:c1", "The user used to live in Delhi.", 1, ["Delhi"], RES),
                           claim("m2:c2", "The user now lives in Bangalore.", 1, ["Bangalore"], RES)])
    assert actions(c) == [CREATE, HISTORICAL, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, None),
                        ("The user used to live in Delhi.", STATUS_HISTORICAL, None, None),
                        ("The user now lives in Bangalore.", STATUS_CURRENT, None, None)]
    assert "past-state restatement of current mem:m1:c1" in c.decisions[1].unresolved
    assert links(c) == [("mem:m2:c2", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user now lives in Bangalore."]


def test_A_unsplit_mixed_claim_is_preserved_without_changing_currentness():
    mixed = "The user used to live in Delhi, but now the user lives in Bangalore."
    c = consolidate(CONV, [DELHI, claim("m2:c1", mixed, 1, ["Delhi", "Bangalore"], RES)])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user lives in Delhi.", STATUS_CURRENT, None, None),
                        (mixed, STATUS_CURRENT, None, None)]
    assert "mixes past-state and present wording" in c.decisions[1].unresolved
    assert links(c) == [] and len(c.sources) == 2


# ---- B. explicit current state -------------------------------------------------------


@pytest.mark.parametrize("old, new, ctype, metadata", [
    ("The user lives in Delhi.", "The user lives in Bangalore now.", RES, None),
    ("The user's team lead is Marco.", "The user's current team lead is Hana.", "team lead",
     {"subject": "the user's team lead"}),
    ("The user prefers Friday meetings.", "The user currently prefers Thursday meetings.", "meeting preference", None),
])
def test_B_explicit_current_state_supersedes_an_established_slot(old, new, ctype, metadata):
    c = consolidate(CONV, [claim("m1:c1", old, 0, claim_type=ctype, metadata=metadata),
                           claim("m2:c1", new, 1, claim_type=ctype, metadata=metadata)])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert table(c) == [(old, STATUS_SUPERSEDED, None, None), (new, STATUS_CURRENT, None, None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == [new] and sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_B_explicit_current_state_without_safe_slot_does_not_supersede():
    c = consolidate(CONV, [claim("m1:c1", "The user's team lead is Marco.", 0, claim_type="team lead"),
                           claim("m2:c1", "The user's current team lead is Hana.", 1, claim_type="team lead")])
    assert actions(c) == [CREATE, CREATE]
    assert [s for _, s, _, _ in table(c)] == [STATUS_CURRENT, STATUS_CURRENT] and links(c) == []


# ---- C / L. future plans -----------------------------------------------------------------


@pytest.mark.parametrize("content, temporal", [
    ("The user will move to Bangalore next month.", "next month"),
    ("The user is going to leave the company in June.", "in June"),
    ("Tomorrow the user will meet Sam.", "Tomorrow"),
])
def test_C_L_future_statement_after_current_state_does_not_supersede(content, temporal):
    c = consolidate(CONV, [DELHI, claim("m2:c1", content, 1, claim_type=RES, temporal=temporal)])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user lives in Delhi.", STATUS_CURRENT, None, None),
                        (content, STATUS_FUTURE, None, None)]
    assert current(c) == ["The user lives in Delhi."]
    assert links(c) == [] and sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}
    assert c.memories["mem:m2:c1"].time_class == TIME_FUTURE


# ---- D. conditional --------------------------------------------------------------------


@pytest.mark.parametrize("content", [
    "If the offer works out, the user will move to Bangalore.",
    "The user might move next month.",
    "The user may switch teams.",
])
def test_D_conditional_statement_never_becomes_current(content):
    c = consolidate(CONV, [DELHI, claim("m2:c1", content, 1, claim_type=RES)])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user lives in Delhi.", STATUS_CURRENT, None, None),
                        (content, STATUS_CONDITIONAL, None, None)]
    assert current(c) == ["The user lives in Delhi."] and links(c) == []
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


# ---- E / F / G / H. dates and validity -------------------------------------------------------


def test_E_F_event_date_differs_from_message_date():
    c = consolidate(CONV, [claim("m1:c1", "The user moved to Bangalore on March 3.", 0, ["Bangalore"],
                                 temporal="on March 3", resolved="2026-03-03", ts="2026-03-10T09:00:00")])
    [m] = c.memories.values()
    assert table(c) == [("The user moved to Bangalore on March 3.", STATUS_CURRENT, "2026-03-03", None)]
    assert (m.event_time, m.created_at) == ("2026-03-03", "2026-03-10T09:00:00")
    assert [(s.claim_id, s.observed_at) for s in c.sources_of(m.memory_id)] == [("m1:c1", "2026-03-10T09:00:00")]
    assert links(c) == [] and current(c) == [m.content]


def test_F_year_precision_is_not_padded_and_observation_is_never_event_time():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, temporal="in 2024", ts="2026-01-15T08:00:00"),
        claim("m2:c1", "The user likes coffee.", 1, ts="2026-01-16T08:00:00")])
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None),
                        ("The user likes coffee.", STATUS_CURRENT, None, None)]
    assert [m.created_at for m in c._ordered()] == ["2026-01-15T08:00:00", "2026-01-16T08:00:00"]
    assert [m.event_time for m in c._ordered()] == ["2024", None]
    assert links(c) == [] and len(c.sources) == 2


def test_G_valid_from_from_start_expressions():
    c = consolidate(CONV, [
        claim("m1:c1", "The user has lived in Pune since 2021.", 0, temporal="since 2021"),
        claim("m2:c1", "The user's gym membership is active starting March 2025.", 1, temporal="starting March 2025"),
        claim("m3:c1", "The user visited Rome in 2019.", 2, temporal="in 2019")])
    assert table(c) == [("The user has lived in Pune since 2021.", STATUS_CURRENT, "2021", None),
                        ("The user's gym membership is active starting March 2025.", STATUS_CURRENT, "2025-03", None),
                        ("The user visited Rome in 2019.", STATUS_CURRENT, None, None)]
    assert c.memories["mem:m3:c1"].event_time == "2019"  # a point in time, not a start
    assert links(c) == [] and len(c.sources) == 3


def test_H_valid_until_only_when_stated():
    c = consolidate(CONV, [
        claim("m1:c1", "The user lived in Delhi from 2020 to 2023.", 0, temporal="from 2020 to 2023"),
        claim("m2:c1", "The user's team lead was Marco until 2023.", 1, temporal="until 2023",
              ts="2026-02-01T10:00:00"),
        claim("m3:c1", "Sam's lease runs until 2023.", 2, temporal="until 2023")])
    assert table(c) == [("The user lived in Delhi from 2020 to 2023.", STATUS_HISTORICAL, "2020", "2023"),
                        ("The user's team lead was Marco until 2023.", STATUS_HISTORICAL, None, "2023"),
                        ("Sam's lease runs until 2023.", STATUS_CURRENT, None, "2023")]  # no observation date
    assert links(c) == [] and len(c.sources) == 3


def test_H_superseding_by_an_undated_claim_sets_no_end_date():
    c = consolidate(CONV, [DELHI, claim("m2:c1", "The user moved to Bangalore.", 1, ["Bangalore"], RES,
                                        ts="2026-05-01T00:00:00")])
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, None),
                        ("The user moved to Bangalore.", STATUS_CURRENT, None, None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]


# ---- I / J. version chains -----------------------------------------------------------------


def test_I_two_version_chain():
    c = consolidate(CONV, [DELHI, claim("m2:c1", "The user moved to Bangalore in 2024.", 1, ["Bangalore"], RES,
                                        temporal="in 2024")])
    assert actions(c) == [CREATE, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, "2024"),
                        ("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Bangalore in 2024."]
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


CHAIN = [claim("m1:c1", "The user lives in Delhi.", 0, ["Delhi"], RES, ts="2023-06-01T00:00:00"),
         claim("m2:c1", "The user moved to Bangalore in 2024.", 1, ["Bangalore"], RES, temporal="in 2024",
               ts="2025-01-10T00:00:00"),
         claim("m3:c1", "The user moved to Mumbai in 2025.", 2, ["Mumbai"], RES, temporal="in 2025",
               ts="2026-01-10T00:00:00")]


def test_J_three_version_chain_is_fully_queryable():
    c = consolidate(CONV, CHAIN)
    assert actions(c) == [CREATE, SUPERSEDE, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, "2024"),
                        ("The user moved to Bangalore in 2024.", STATUS_SUPERSEDED, "2024", "2025"),
                        ("The user moved to Mumbai in 2025.", STATUS_CURRENT, "2025", None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES"), ("mem:m3:c1", "mem:m2:c1", "SUPERSEDES")]
    assert [m.memory_id for m in c.history("mem:m3:c1")] == ["mem:m2:c1", "mem:m1:c1"]
    assert current(c) == ["The user moved to Mumbai in 2025."]
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"], "mem:m3:c1": ["m3:c1"]}


def test_J_historical_statement_between_updates_is_untouched():
    c = consolidate(CONV, [DELHI,
                           claim("m2:c1", "The user used to live in Chennai.", 1, ["Chennai"], RES),
                           claim("m3:c1", "The user moved to Bangalore in 2024.", 2, ["Bangalore"], RES,
                                 temporal="in 2024")])
    assert actions(c) == [CREATE, HISTORICAL, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, "2024"),
                        ("The user used to live in Chennai.", STATUS_HISTORICAL, None, None),
                        ("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None)]
    assert links(c) == [("mem:m3:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Bangalore in 2024."] and len(c.sources) == 3


def test_J_future_statement_between_updates_is_untouched():
    c = consolidate(CONV, [DELHI,
                           claim("m2:c1", "The user will move to Pune next month.", 1, ["Pune"], RES,
                                 temporal="next month"),
                           claim("m3:c1", "The user moved to Bangalore in 2024.", 2, ["Bangalore"], RES,
                                 temporal="in 2024")])
    assert actions(c) == [CREATE, CREATE, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, "2024"),
                        ("The user will move to Pune next month.", STATUS_FUTURE, None, None),
                        ("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None)]
    assert links(c) == [("mem:m3:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Bangalore in 2024."] and len(c.sources) == 3


def test_J_duplicate_restatement_between_updates_adds_a_source_only():
    c = consolidate(CONV, [CHAIN[0], CHAIN[1],
                           claim("m2b:c1", "The user moved to Bangalore in 2024.", 2, ["Bangalore"], RES,
                                 temporal="in 2024", ts="2025-06-01T00:00:00"),
                           claim("m3:c1", "The user moved to Mumbai in 2025.", 3, ["Mumbai"], RES,
                                 temporal="in 2025", ts="2026-01-10T00:00:00")])
    assert actions(c) == [CREATE, SUPERSEDE, KEEP, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, "2024"),
                        ("The user moved to Bangalore in 2024.", STATUS_SUPERSEDED, "2024", "2025"),
                        ("The user moved to Mumbai in 2025.", STATUS_CURRENT, "2025", None)]
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1", "m2b:c1"], "mem:m3:c1": ["m3:c1"]}
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES"), ("mem:m3:c1", "mem:m2:c1", "SUPERSEDES")]


# ---- K / Q / S. restatements ----------------------------------------------------------------


def test_K_historical_restatement_adds_provenance_without_reactivating():
    c = consolidate(CONV, [DELHI,
                           claim("m2:c1", "The user moved to Bangalore.", 1, ["Bangalore"], RES),
                           claim("m3:c1", "The user used to live in Delhi.", 2, ["Delhi"], RES)])
    assert actions(c) == [CREATE, SUPERSEDE, KEEP]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, None),
                        ("The user moved to Bangalore.", STATUS_CURRENT, None, None)]
    assert sources(c) == {"mem:m1:c1": ["m1:c1", "m3:c1"], "mem:m2:c1": ["m2:c1"]}
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Bangalore."]
    assert "status unchanged" in c.decisions[2].reason


def test_Q_historical_then_current_then_historical_restatement():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0, ["Delhi"], RES),
                           claim("m2:c1", "The user lives in Bangalore now.", 1, ["Bangalore"], RES),
                           claim("m3:c1", "The user used to live in Delhi.", 2, ["Delhi"], RES)])
    assert actions(c) == [HISTORICAL, CREATE, KEEP]
    assert table(c) == [("The user used to live in Delhi.", STATUS_HISTORICAL, None, None),
                        ("The user lives in Bangalore now.", STATUS_CURRENT, None, None)]
    assert sources(c) == {"mem:m1:c1": ["m1:c1", "m3:c1"], "mem:m2:c1": ["m2:c1"]}
    assert links(c) == [] and current(c) == ["The user lives in Bangalore now."]


def test_S_duplicate_historical_statement():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0, ["Delhi"]),
                           claim("m4:c1", "The user used to live in Delhi.", 3, ["Delhi"])])
    assert actions(c) == [HISTORICAL, KEEP]
    assert table(c) == [("The user used to live in Delhi.", STATUS_HISTORICAL, None, None)]
    assert sources(c) == {"mem:m1:c1": ["m1:c1", "m4:c1"]}
    assert links(c) == [] and current(c) == []


# ---- M. out-of-order event dates ---------------------------------------------------------------


def test_M_later_observed_but_earlier_dated_move_becomes_the_older_version():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, ["Bangalore"], RES, temporal="in 2024",
              ts="2025-01-01T00:00:00"),
        claim("m2:c1", "The user moved to Mumbai in 2023.", 1, ["Mumbai"], RES, temporal="in 2023",
              ts="2026-01-01T00:00:00")])
    assert actions(c) == [CREATE, HISTORICAL]
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None),
                        ("The user moved to Mumbai in 2023.", STATUS_SUPERSEDED, "2023", "2024")]
    assert links(c) == [("mem:m1:c1", "mem:m2:c1", "SUPERSEDES")]  # newer by event time -> older
    assert current(c) == ["The user moved to Bangalore in 2024."]
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_M_later_dated_move_supersedes_and_later_historical_claim_does_not():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, ["Bangalore"], RES, temporal="in 2024",
              ts="2025-01-01T00:00:00"),
        claim("m2:c1", "The user moved to Delhi in 2025.", 1, ["Delhi"], RES, temporal="in 2025",
              ts="2026-01-01T00:00:00"),
        claim("m3:c1", "The user used to live in Bangalore in 2024.", 2, ["Bangalore"], RES, temporal="in 2024",
              ts="2026-02-01T00:00:00")])
    assert actions(c) == [CREATE, SUPERSEDE, HISTORICAL]
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_SUPERSEDED, "2024", "2025"),
                        ("The user moved to Delhi in 2025.", STATUS_CURRENT, "2025", None),
                        ("The user used to live in Bangalore in 2024.", STATUS_HISTORICAL, None, None)]
    assert c.memories["mem:m3:c1"].event_time == "2024"
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Delhi in 2025."] and len(c.sources) == 3


def test_M_overlapping_dates_preserve_ambiguity():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, ["Bangalore"], RES, temporal="in 2024"),
        claim("m2:c1", "The user moved to Mumbai in March 2024.", 1, ["Mumbai"], RES, temporal="in March 2024")])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None),
                        ("The user moved to Mumbai in March 2024.", STATUS_CURRENT, "2024-03", None)]
    assert "event order relative to mem:m1:c1 cannot be established" in c.decisions[1].unresolved
    assert links(c) == [] and len(c.sources) == 2


def test_M_dated_move_before_an_observed_present_state_is_not_ordered():
    c = consolidate(CONV, [
        claim("m1:c1", "The user lives in Delhi.", 0, ["Delhi"], RES, ts="2026-01-10T00:00:00"),
        claim("m2:c1", "The user moved to Bangalore in 2024.", 1, ["Bangalore"], RES, temporal="in 2024",
              ts="2026-02-01T00:00:00")])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user lives in Delhi.", STATUS_CURRENT, None, None),
                        ("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None)]
    assert c.decisions[1].unresolved and links(c) == []


def test_M_out_of_order_without_a_safe_slot_keeps_both_current():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, ["Bangalore"], temporal="in 2024"),
        claim("m2:c1", "The user moved to Mumbai in 2023.", 1, ["Mumbai"], temporal="in 2023")])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_CURRENT, "2024", None),
                        ("The user moved to Mumbai in 2023.", STATUS_CURRENT, "2023", None)]
    assert links(c) == [] and len(c.sources) == 2


# ---- N. contradictory dates ----------------------------------------------------------------------


def test_N_same_change_with_conflicting_dates_is_contested_not_resolved_by_order():
    c = consolidate(CONV, [
        claim("m1:c1", "The user moved to Bangalore in 2024.", 0, ["Bangalore"], RES, temporal="in 2024",
              ts="2025-01-01T00:00:00"),
        claim("m2:c1", "The user moved to Bangalore in 2023.", 1, ["Bangalore"], RES, temporal="in 2023",
              ts="2026-01-01T00:00:00")])
    assert actions(c) == [CREATE, CONTRADICTION]
    assert table(c) == [("The user moved to Bangalore in 2024.", STATUS_CONTESTED, "2024", None),
                        ("The user moved to Bangalore in 2023.", STATUS_CONTESTED, "2023", None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "CONTRADICTS")]
    assert len(current(c)) == 2  # no winner
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_N_explicit_contradicts_with_dates_remains_contested():
    c = consolidate(CONV, [
        claim("m1:c1", "The museum closed in 2019.", 0, ["museum"], temporal="in 2019"),
        claim("m2:c1", "The museum did not close in 2019.", 1, ["museum"], polarity="negative", temporal="in 2019")],
        [rel("m2:c1", "m1:c1", "CONTRADICTS")])
    assert actions(c) == [CREATE, CONTRADICTION]
    assert [s for _, s, _, _ in table(c)] == [STATUS_CONTESTED, STATUS_CONTESTED]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "CONTRADICTS")] and len(c.sources) == 2


# ---- O. ambiguous temporal wording ------------------------------------------------------------


def test_O_ambiguous_past_tense_is_preserved_and_never_used_for_transitions():
    c = consolidate(CONV, [claim("m1:c1", "The user lives in Bangalore.", 0, ["Bangalore"], RES),
                           claim("m2:c1", "The user was in Delhi.", 1, ["Delhi"], RES),
                           claim("m3:c1", "The user moved to Pune.", 2, ["Pune"], RES)])
    assert actions(c) == [CREATE, CREATE, SUPERSEDE]
    assert table(c) == [("The user lives in Bangalore.", STATUS_SUPERSEDED, None, None),
                        ("The user was in Delhi.", STATUS_CURRENT, None, None),
                        ("The user moved to Pune.", STATUS_CURRENT, None, None)]
    assert "past tense without explicit past-state wording" in c.decisions[1].unresolved
    assert c.memories["mem:m2:c1"].time_class == TIME_AMBIGUOUS_PAST
    assert links(c) == [("mem:m3:c1", "mem:m1:c1", "SUPERSEDES")]  # the ambiguous memory is not a slot target
    assert len(c.sources) == 3


# ---- P. multiple historical memories --------------------------------------------------------


def test_P_multiple_historical_memories_coexist_with_the_current_one():
    c = consolidate(CONV, [claim("m1:c1", "The user used to live in Delhi.", 0, ["Delhi"], RES),
                           claim("m2:c1", "The user used to live in Chennai.", 1, ["Chennai"], RES),
                           claim("m3:c1", "The user lives in Bangalore now.", 2, ["Bangalore"], RES)])
    assert actions(c) == [HISTORICAL, HISTORICAL, CREATE]
    assert table(c) == [("The user used to live in Delhi.", STATUS_HISTORICAL, None, None),
                        ("The user used to live in Chennai.", STATUS_HISTORICAL, None, None),
                        ("The user lives in Bangalore now.", STATUS_CURRENT, None, None)]
    assert links(c) == [] and current(c) == ["The user lives in Bangalore now."] and len(c.sources) == 3


# ---- R. current -> future -> current -----------------------------------------------------------


def test_R_current_future_current():
    c = consolidate(CONV, [DELHI,
                           claim("m2:c1", "The user will move to Bangalore next month.", 1, ["Bangalore"], RES,
                                 temporal="next month"),
                           claim("m3:c1", "The user moved to Bangalore.", 2, ["Bangalore"], RES)])
    assert actions(c) == [CREATE, CREATE, SUPERSEDE]
    assert table(c) == [("The user lives in Delhi.", STATUS_SUPERSEDED, None, None),
                        ("The user will move to Bangalore next month.", STATUS_FUTURE, None, None),
                        ("The user moved to Bangalore.", STATUS_CURRENT, None, None)]
    assert links(c) == [("mem:m3:c1", "mem:m1:c1", "SUPERSEDES")]
    assert current(c) == ["The user moved to Bangalore."] and len(c.sources) == 3


# ---- T. explicit UPDATES + temporal metadata ----------------------------------------------------


def test_T_explicit_updates_with_dates_sets_validity_bounds():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid since 2021.", 0, ["Leah", "Madrid"],
                                 temporal="since 2021"),
                           claim("m2:c1", "Leah moved to Valencia in March 2025.", 1, ["Leah", "Valencia"],
                                 temporal="in March 2025")],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, UPDATE]
    assert table(c) == [("Leah lives in Madrid since 2021.", STATUS_SUPERSEDED, "2021", "2025-03"),
                        ("Leah moved to Valencia in March 2025.", STATUS_CURRENT, "2025-03", None)]
    assert [(l.type, l.relation_id) for l in c.links] == [("UPDATES", "m2:r1")]
    assert current(c) == ["Leah moved to Valencia in March 2025."]
    assert sources(c) == {"mem:m1:c1": ["m1:c1"], "mem:m2:c1": ["m2:c1"]}


def test_T_explicit_updates_from_a_future_claim_links_without_superseding():
    c = consolidate(CONV, [claim("m1:c1", "Leah lives in Madrid.", 0, ["Leah", "Madrid"]),
                           claim("m2:c1", "Leah will move to Valencia next month.", 1, ["Leah", "Valencia"],
                                 temporal="next month")],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("Leah lives in Madrid.", STATUS_CURRENT, None, None),
                        ("Leah will move to Valencia next month.", STATUS_FUTURE, None, None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "UPDATES")]
    assert "future claim does not supersede" in c.decisions[1].unresolved
    assert current(c) == ["Leah lives in Madrid."] and len(c.sources) == 2


def test_T_explicit_updates_contradicted_by_event_dates_does_not_supersede():
    c = consolidate(CONV, [claim("m1:c1", "Leah moved to Madrid in 2024.", 0, ["Leah", "Madrid"], temporal="in 2024"),
                           claim("m2:c1", "Leah moved to Valencia in 2022.", 1, ["Leah", "Valencia"],
                                 temporal="in 2022")],
                    [rel("m2:c1", "m1:c1", "UPDATES")])
    assert actions(c) == [CREATE, CREATE]
    assert table(c) == [("Leah moved to Madrid in 2024.", STATUS_CURRENT, "2024", None),
                        ("Leah moved to Valencia in 2022.", STATUS_CURRENT, "2022", None)]
    assert links(c) == [("mem:m2:c1", "mem:m1:c1", "UPDATES")]
    assert "stated dates do not place it after mem:m1:c1" in c.decisions[1].unresolved
    assert len(c.sources) == 2


# ---- determinism and persistence ---------------------------------------------------------------


STREAM = [*CHAIN,
          claim("m4:c1", "The user used to live in Delhi.", 3, ["Delhi"], RES),
          claim("m5:c1", "The user will move to Goa next year.", 4, ["Goa"], RES, temporal="next year"),
          claim("m6:c1", "The user moved to Pune in 2022.", 5, ["Pune"], RES, temporal="in 2022"),
          claim("m7:c1", "The user was in Delhi.", 6, ["Delhi"], RES),
          claim("m8:c1", "Leah lives in Madrid since 2021.", 7, ["Leah"], temporal="since 2021"),
          claim("m9:c1", "Leah moved to Valencia in March 2025.", 8, ["Leah"], temporal="in March 2025")]
STREAM_RELATIONS = [rel("m9:c1", "m8:c1", "UPDATES")]


def _snapshot(c: Consolidator):
    return ([m.to_dict() for m in c.memories.values()], [s.to_dict() for s in c.sources],
            [l.to_dict() for l in c.links], [d.to_dict() for d in c.decisions])


def test_temporal_consolidation_is_deterministic_regardless_of_input_order():
    expected = _snapshot(consolidate(CONV, STREAM, STREAM_RELATIONS))
    for seed in range(5):
        claims = STREAM[:]
        random.Random(seed).shuffle(claims)
        assert _snapshot(consolidate(CONV, claims, STREAM_RELATIONS)) == expected
    c = consolidate(CONV, STREAM, STREAM_RELATIONS)
    assert current(c) == ["The user moved to Mumbai in 2025.", "The user was in Delhi.",
                          "Leah moved to Valencia in March 2025."]
    assert len(c.memories) == len(STREAM) - 1  # "used to live in Delhi" adds provenance to Delhi


def test_temporal_fields_round_trip_through_the_store(store):
    messages = [{"id": x.provenance.message_id, "role": "user", "content": x.content,
                 "timestamp": x.provenance.timestamp} for x in STREAM]
    store.save_conversation(parse_conversation({"conversation_id": CONV, "messages": messages}), [],
                            np.zeros((0, 0), dtype=np.float32), "none")
    c = consolidate(CONV, STREAM, STREAM_RELATIONS)
    ms = MemoryStore(store.connection)
    ms.save(c)
    assert [m.to_dict() for m in ms.get_memories(CONV)] == [m.to_dict() for m in c._ordered()]
    assert {m.time_class for m in ms.get_memories(CONV)} >= {TIME_AMBIGUOUS_PAST, TIME_FUTURE, TIME_TRANSITION}
    assert {(m.status, m.event_time) for m in ms.get_memories(CONV) if m.time_class == TIME_FUTURE} == {
        (STATUS_FUTURE, None)}
