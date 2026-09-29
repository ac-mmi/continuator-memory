"""State tracking and temporal validity: consolidation rules and their persistence."""

import numpy as np
import pytest

from continuator_memory.memory.consolidation import decide_memory
from continuator_memory.memory.extraction import ExtractedMemory, MemoryExtractor
from continuator_memory.memory.formation import MemoryFormation
from continuator_memory.memory.models import Action, Memory
from continuator_memory.memory.store import MemoryStore
from continuator_memory.models import parse_conversation
from tests.helpers import ScriptedJSONClient, doc, mem

MESSAGES = [
    ("m1", "user", "Project Nova launches on June 3."),
    ("m2", "user", "Nova's launch moved to June 17."),
    ("m3", "user", "We discussed the June 3 date with finance."),
    ("m4", "assistant", "Noted, the launch is on June 17."),
    ("m5", "user", "Nova also has a budget of 40k."),
    ("m6", "user", "Nova's launch is now July 1."),
]
SCRIPT = {
    MESSAGES[0][2]: doc([mem("Project Nova", "launch_date", "June 3")]),
    MESSAGES[1][2]: doc([mem("Nova", "launch_date", "June 17", "update", "June 3")]),
    MESSAGES[2][2]: doc([mem("Project Nova", "launch_date", "June 3", "reference")]),
    MESSAGES[3][2]: doc([mem("Project Nova", "launch date", "June 17th")]),
    MESSAGES[4][2]: doc([mem("Project Nova", "budget", "40k")]),
    MESSAGES[5][2]: doc([mem("Project Nova", "launch_date", "July 1", "update")]),
}


def _conversation(n):
    return parse_conversation(
        {
            "conversation_id": "c1",
            "messages": [
                {"id": mid, "role": role, "content": text, "timestamp": f"2026-01-0{i + 1}"}
                for i, (mid, role, text) in enumerate(MESSAGES[:n])
            ],
        }
    )


def _form(store, n):
    conv = _conversation(n)
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    ms = MemoryStore(store.connection)
    formation = MemoryFormation(ms, MemoryExtractor(ScriptedJSONClient(SCRIPT)), clock=lambda: "T")
    report = formation.form(conv)
    return ms, report


def _slot(ms, predicate="launch_date"):
    return {m.value: m for m in ms.memories("c1", predicate=predicate)}


def test_new_fact_creates_memory(store):
    ms, report = _form(store, 1)
    (m,) = ms.memories("c1")
    assert (m.subject, m.predicate, m.value) == ("Project Nova", "launch_date", "June 3")
    assert m.is_current and m.supersedes_memory_id is None
    assert m.source_message_ids == ("m1",)
    assert m.valid_from == "2026-01-01" and m.valid_until is None
    assert report.actions == {"memory:CREATE": 1}


def test_new_value_supersedes_old_value(store):
    ms, _ = _form(store, 2)
    slot = _slot(ms)
    assert slot["June 17"].supersedes_memory_id == slot["June 3"].memory_id
    assert slot["June 3"].valid_until == "2026-01-02"


def test_old_value_remains_stored_and_current_is_marked(store):
    ms, _ = _form(store, 2)
    slot = _slot(ms)
    assert set(slot) == {"June 3", "June 17"}
    assert slot["June 3"].is_current is False
    assert slot["June 17"].is_current is True
    assert [m.value for m in ms.memories("c1", current_only=True)] == ["June 17"]


def test_historical_value_can_still_be_retrieved(store):
    ms, _ = _form(store, 6)
    history = [m.value for m in ms.memories("c1", predicate="launch_date") if not m.is_current]
    assert history == ["June 3", "June 17"]
    chain, current = [], ms.memories("c1", predicate="launch_date", current_only=True)[0]
    while current is not None:
        chain.append(current.value)
        current = ms.get_memory(current.supersedes_memory_id) if current.supersedes_memory_id else None
    assert chain == ["July 1", "June 17", "June 3"]


def test_reference_and_unrelated_values_do_not_supersede(store):
    ms, report = _form(store, 5)
    slot = _slot(ms)
    assert slot["June 17"].is_current  # m3 only referenced June 3
    assert "memory:IGNORE" in report.actions
    assert _slot(ms, "budget")["40k"].is_current  # different slot: independent
    assert len(ms.memories("c1", current_only=True)) == 2


def test_duplicate_fact_does_not_create_duplicate_current_memory(store):
    ms, report = _form(store, 4)
    current = ms.memories("c1", predicate="launch_date", current_only=True)
    assert len(current) == 1
    assert current[0].source_message_ids == ("m2", "m4")
    assert report.actions["memory:UPDATE"] == 1


def test_consolidation_log_records_every_decision(store):
    ms, _ = _form(store, 6)
    rows = store.connection.execute(
        "SELECT message_id, action FROM consolidation_log ORDER BY log_id"
    ).fetchall()
    assert rows == [
        ("m1", "CREATE"), ("m2", "SUPERSEDE"), ("m3", "IGNORE"),
        ("m4", "UPDATE"), ("m5", "CREATE"), ("m6", "SUPERSEDE"),
    ]
    assert ms.stats("c1")["memories"] == 4


def test_reingest_rebuilds_memory(store):
    _form(store, 6)
    ms, _ = _form(store, 1)
    assert [m.value for m in ms.memories("c1")] == ["June 3"]


# ---- pure decision rules ----------------------------------------------------


def _m(mid, value, current=True, sources=("m1",), pos=0):
    return Memory(mid, "c", "e", "S", "p", value, "state", sources, pos, "T", None, None,
                  current, None)


@pytest.mark.parametrize(
    "candidate, slot, action",
    [
        (ExtractedMemory("S", "p", "A"), [], Action.CREATE),
        (ExtractedMemory("S", "p", "A"), [_m("x", "A")], Action.UPDATE),
        (ExtractedMemory("S", "p", "a."), [_m("x", "A", sources=("m9",))], Action.KEEP),
        (ExtractedMemory("S", "p", "B"), [_m("x", "A")], Action.CREATE),
        (ExtractedMemory("S", "p", "B", "update"), [_m("x", "A")], Action.SUPERSEDE),
        (ExtractedMemory("S", "p", "B", "correction"), [_m("x", "A")], Action.SUPERSEDE),
        (ExtractedMemory("S", "p", "B", "reference"), [_m("x", "A")], Action.IGNORE),
        (ExtractedMemory("S", "p", "A"), [_m("x", "A", current=False), _m("y", "B")],
         Action.IGNORE),
        (ExtractedMemory("S", "p", "B", "update"), [], Action.CREATE),
    ],
)
def test_decide_memory(candidate, slot, action):
    assert decide_memory(candidate, slot, "m9").action is action


def test_state_with_new_value_keeps_existing_values_current():
    d = decide_memory(ExtractedMemory("S", "p", "B"), [_m("x", "A")], "m9")
    assert d.action is Action.CREATE and d.supersede_ids == ()


def test_update_naming_previous_value_only_supersedes_that_value():
    slot = [_m("x", "tea"), _m("y", "coffee")]
    d = decide_memory(ExtractedMemory("S", "p", "juice", "update", "coffee"), slot, "m9")
    assert d.supersede_ids == ("y",)


def test_update_from_unrecorded_value_records_history():
    d = decide_memory(ExtractedMemory("S", "p", "July 1", "update", "June 17"), [], "m9")
    assert d.action is Action.SUPERSEDE and d.create_previous == "June 17"


def test_change_from_unrecorded_value_is_persisted_as_history(store):
    text = "The launch was moved from June 17 to July 1."
    conv = parse_conversation(
        {"conversation_id": "c1", "messages": [{"id": "m1", "role": "user", "content": text}]}
    )
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    ms = MemoryStore(store.connection)
    client = ScriptedJSONClient({text: doc([mem("launch", "date", "July 1", "update", "June 17")])})
    MemoryFormation(ms, MemoryExtractor(client), clock=lambda: "T").form(conv)
    slot = _slot(ms, "date")
    assert slot["June 17"].is_current is False
    assert slot["July 1"].is_current and slot["July 1"].supersedes_memory_id == slot["June 17"].memory_id


def test_state_naming_current_value_as_previous_is_a_change():
    d = decide_memory(ExtractedMemory("S", "p", "B", "state", "A"), [_m("x", "A")], "m9")
    assert d.action is Action.SUPERSEDE and d.supersede_ids == ("x",)


def test_reversion_to_superseded_value_needs_replaced_value_in_message():
    slot = [_m("x", "library", current=False), _m("y", "cafe")]
    cand = ExtractedMemory("S", "p", "library", "update", "cafe")
    assert decide_memory(cand, slot, "m9", "We discussed the old library venue.").action is (
        Action.IGNORE
    )
    assert decide_memory(cand, slot, "m9", "Back to the library instead of the cafe.").action is (
        Action.SUPERSEDE
    )


def _form_texts(store, script, texts):
    conv = parse_conversation({"conversation_id": "c1", "messages": [
        {"id": f"m{i + 1}", "role": "user", "content": t} for i, t in enumerate(texts)]})
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    ms = MemoryStore(store.connection)
    report = MemoryFormation(ms, MemoryExtractor(ScriptedJSONClient(script)),
                             clock=lambda: "T").form(conv)
    return ms, report


def test_update_attached_to_wrong_subject_resolves_to_the_changed_slot(store):
    texts = ["The book club meets at the library.",
             "The library is closing, so the meeting moves to the Greenway Cafe."]
    script = {
        texts[0]: doc([mem("book club", "meeting_location", "library")]),
        texts[1]: doc([mem("library", "meeting_location", "Greenway Cafe", "update", "library")]),
    }
    ms, _ = _form_texts(store, script, texts)
    slot = {m.value: m for m in ms.memories("c1", predicate="meeting_location")}
    assert slot["Greenway Cafe"].subject == "book club" and slot["Greenway Cafe"].is_current
    assert slot["library"].is_current is False


def test_fact_whose_value_is_a_known_entity_becomes_a_relationship(store):
    texts = ["Dana is a retired chemistry teacher.", "Dana keeps the reading list for the club.",
             "The club is run by Dana."]
    script = {
        texts[0]: {"facts": [mem("Dana", "occupation", "retired chemistry teacher")]},
        texts[1]: {"facts": [mem("club", "reading_list_keeper", "Dana")]},
        texts[2]: {"facts": [mem("club", "run_by", "Dana")]},
    }
    ms, report = _form_texts(store, script, texts)
    assert [m.predicate for m in ms.memories("c1")] == ["occupation"]
    assert [(r.subject, r.predicate, r.object) for r in ms.relationships("c1")] == [
        ("club", "reading_list_keeper", "Dana"), ("club", "run_by", "Dana")
    ]
    assert report.actions == {"memory:CREATE": 1, "relationship:CREATE": 2}
