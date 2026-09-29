"""Deterministic extraction provenance: DIRECT / STATED_EARLIER / CONTEXTUAL, else rejected."""

import json

import numpy as np
import pytest

from continuator_memory.memory.extraction import (
    CONTEXTUAL,
    COPIED,
    DIRECT,
    STATED_EARLIER,
    UNSUPPORTED,
    ExtractedMemory,
    ExtractionResult,
    KnownFact,
    MemoryExtractor,
    classify_value,
    ground,
)
from continuator_memory.memory.formation import MemoryFormation
from continuator_memory.memory.store import MemoryStore
from continuator_memory.models import Message, parse_conversation
from tests.helpers import ScriptedJSONClient, mem


def _msg(message_id, content, role="user", position=0):
    return Message(message_id, "c1", position, role, content)


def _ground_one(value, message, recent=(), known=(), subject="user", predicate="p"):
    return ground(ExtractionResult(memories=[ExtractedMemory(subject, predicate, value)]),
                  message, recent, known)


FLIGHT_M01 = "I'm visiting Spain next month. My flight to Madrid is on the 9th."
FLIGHT_M11 = "Correction: I misread my ticket. The flight is actually on the 11th, not the 9th."
FILLER = ["I'm also allergic to penicillin.", "Worth declaring that on your insurance form.",
          "Noted the Wi-Fi password."]
LEAH_STAY = "My sister Leah lives in Madrid, so I'll stay at Leah's place."
LEAH_FIX = ("Also I mixed up my family: Leah lives in Valencia. It's my cousin Tom who lives "
            "in Madrid, so I'll stay with Tom.")
PEPPER = "I have a dog too, a three-year-old border collie called Pepper."
COLLIES = "Border collies need a lot of exercise and mental stimulation."
GARDEN = "I want to start a vegetable garden this spring. I'm thinking of building three raised beds."


# ---- DIRECT -----------------------------------------------------------------------------

@pytest.mark.parametrize("value, message", [
    ("backend developer", "Omar is the backend developer."),
    ("FALAR BEM", "I'm learning Portuguese at a school called Falar Bem."),
    ("harbor 72", "It's harbor-72, not harbour-27."),
    ("55,000 euros", "The budget is now 55000 euros."),
    ("June 17", "The launch moves to June 17th."),
    ("Greenway cafe", "The meeting moves to the Greenway Café."),
])
def test_direct_value_uses_current_message_as_evidence(value, message):
    (m,) = _ground_one(value, _msg("m5", message)).memories
    assert (m.grounding, m.evidence_message_ids) == (DIRECT, ("m5",))


# ---- CONTEXTUAL -------------------------------------------------------------------------

def test_contextual_update_inherits_qualifier_from_known_fact():
    known = [KnownFact("user", "departure_date", "9th next month", ("m01",))]
    recent = [_msg(f"m{8 + i:02d}", t) for i, t in enumerate(FILLER)]
    grounding, evidence = classify_value("11th next month", _msg("m11", FLIGHT_M11), recent, known)
    assert grounding == CONTEXTUAL
    assert evidence == ("m11", "m01")


def test_context_recombination_without_new_message_information_is_rejected():
    known = [KnownFact("team", "meeting_place", "Leeds office", ("m1",))]
    recent = [_msg("m2", "York is lovely in spring.")]
    message = _msg("m3", "The York trip was fun.")
    assert classify_value("Leeds office York", message, recent, known) == (None, ())
    out = _ground_one("Leeds office York", message, recent, known, subject="team")
    assert out.memories == [] and out.grounding[0]["grounding"] == UNSUPPORTED


# ---- COPIED / UNSUPPORTED ---------------------------------------------------------------

def test_copied_known_fact_is_dropped():
    known = [KnownFact("user", "accommodation", "Leah's place in Madrid", ("m03",))]
    out = _ground_one("Leah's place in Madrid", _msg("m13", LEAH_FIX), [], known,
                      predicate="accommodation")
    assert out.memories == []
    assert out.rejected == ["memory 'user'.accommodation: copied known fact"]
    assert out.grounding == [{"subject": "user", "predicate": "accommodation",
                              "value": "Leah's place in Madrid", "grounding": COPIED,
                              "evidence": []}]


def test_value_with_ungrounded_word_is_unsupported():
    out = _ground_one("bytes raised beds", _msg("m01", GARDEN), predicate="number_of_beds")
    assert out.memories == []
    assert out.rejected == ["memory 'user'.number_of_beds: value not grounded"]
    assert out.grounding[0]["grounding"] == UNSUPPORTED


# ---- STATED_EARLIER ---------------------------------------------------------------------

def test_value_stated_in_earlier_message_is_attributed_to_it():
    recent = [_msg("m06", "The bus is a relaxing option.", "assistant"), _msg("m07", PEPPER)]
    value = "a three-year-old border collie called Pepper"
    assert classify_value(value, _msg("m08", COLLIES, "assistant"), recent, []) == (
        STATED_EARLIER, ("m07",))


# ---- unchanged rules and determinism ------------------------------------------------------

def test_rules_unrelated_to_value_classification_are_unchanged():
    result = ExtractionResult(memories=[
        ExtractedMemory("Zed", "role", "backend developer"),
        ExtractedMemory("Omar", "role", "backend developer", "update", "chef"),
        ExtractedMemory("Omar", "role", "backend developer", "update", "developer"),
    ], rejected=["facts[3]: missing or empty 'value'"])
    out = ground(result, _msg("m2", "Omar is the backend developer."),
                 known=[KnownFact("Mei", "role", "chef", ("m1",))])
    assert out.rejected == ["facts[3]: missing or empty 'value'",
                            "memory 'Zed': subject not grounded"]
    assert [m.previous_value for m in out.memories] == ["chef", "developer"]


def test_classification_is_deterministic():
    known = [KnownFact("user", "departure_date", "9th next month", ("m01", "m02"))]
    recent = [_msg("m09", PEPPER), _msg("m10", GARDEN)]
    result = ExtractionResult(memories=[
        ExtractedMemory("user", "departure_date", "11th next month", "update"),
        ExtractedMemory("user", "has_dog", "border collie called Pepper"),
        ExtractedMemory("user", "number_of_beds", "bytes raised beds"),
    ])
    runs = [ground(result, _msg("m11", FLIGHT_M11), recent, known) for _ in range(3)]
    assert all(r == runs[0] for r in runs)
    assert [(g["grounding"], g["evidence"]) for g in runs[0].grounding] == [
        (CONTEXTUAL, ["m11", "m01", "m02"]), (STATED_EARLIER, ["m09"]), (UNSUPPORTED, []),
    ]


# ---- end-to-end formation ---------------------------------------------------------------

def _form(store, texts, script, roles=None):
    roles = roles or ["user"] * len(texts)
    conv = parse_conversation({"conversation_id": "c1", "messages": [
        {"id": f"m{i + 1:02d}", "role": r, "content": t}
        for i, (r, t) in enumerate(zip(roles, texts))]})
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    ms = MemoryStore(store.connection)
    report = MemoryFormation(ms, MemoryExtractor(ScriptedJSONClient(script)),
                             clock=lambda: "T").form(conv)
    return ms, report


def test_formation_contextual_update_replaces_stale_value(store):
    texts = [FLIGHT_M01, *FILLER, FLIGHT_M11]
    script = {
        FLIGHT_M01: {"facts": [mem("user", "departure_date", "9th next month")]},
        FLIGHT_M11: {"facts": [mem("user", "departure_date", "11th next month", "update",
                                   "9th next month")]},
    }
    ms, report = _form(store, texts, script)
    slot = {m.value: m for m in ms.memories("c1", predicate="departure_date")}
    assert slot["11th next month"].is_current and not slot["9th next month"].is_current
    assert slot["11th next month"].source_message_ids == ("m05",)
    assert report.grounding == {DIRECT: 1, CONTEXTUAL: 1}
    logged = json.loads(store.connection.execute(
        "SELECT grounding_json FROM extraction_log WHERE message_id = 'm05'").fetchone()[0])
    assert logged[0]["grounding"] == CONTEXTUAL and logged[0]["evidence"] == ["m05", "m01"]


def test_formation_copied_fact_does_not_reinforce_stale_value(store):
    texts = [LEAH_STAY, *FILLER, LEAH_FIX]
    stale = mem("user", "accommodation", "Leah's place in Madrid")
    script = {LEAH_STAY: {"facts": [stale]},
              LEAH_FIX: {"facts": [dict(stale, kind="update")]}}
    ms, report = _form(store, texts, script)
    (m,) = ms.memories("c1", predicate="accommodation")
    assert m.source_message_ids == ("m01",)
    actions = store.connection.execute(
        "SELECT action FROM consolidation_log WHERE message_id = 'm05'").fetchall()
    assert actions == []
    assert report.grounding == {DIRECT: 1, COPIED: 1}


def test_formation_attributes_late_extraction_to_the_stating_message(store):
    texts = ["Hi, I'm Sam and I just moved to Braga.", PEPPER, COLLIES]
    script = {
        texts[0]: {"facts": [mem("Sam", "location", "Braga")]},
        COLLIES: {"facts": [mem("Sam", "has_dog", "a three-year-old border collie called Pepper")]},
    }
    ms, report = _form(store, texts, script, roles=["user", "user", "assistant"])
    (dog,) = ms.memories("c1", predicate="has_dog")
    assert dog.source_message_ids == ("m02",)
    assert dog.observed_position == 1
    assert report.grounding == {DIRECT: 1, STATED_EARLIER: 1}
