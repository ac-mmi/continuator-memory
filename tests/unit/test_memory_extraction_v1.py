"""V1 memory extraction: contract, validation, consolidation behaviour, provenance and retrieval compatibility.

The model is replaced by a scripted JSON client; everything else is the production path:
ingest_imported(extract="memories") -> V2IngestionPipeline -> validation -> SemanticStore -> consolidation.
"""

from __future__ import annotations

import json

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import MEMORY_PROTOCOL, AtomizationRequestError, SemanticAtomizer
from continuator_memory.v2.memory_prompt import MEMORY_SYSTEM_PROMPT
from continuator_memory.v2.memory_records import memory_records
from continuator_memory.v2.store import SemanticStore
from continuator_memory.webapi.extract import ingest_imported
from tests.helpers import HashingEmbedder

CID = "imported"


class ScriptedClient:
    """Returns a scripted entry per target message; anything not scripted is 'no_durable_assertion'."""

    model_name = "scripted"

    def __init__(self, entries=None, raw_by_window=None, fail_windows=()):
        self.entries = entries or {}
        self.raw_by_window = raw_by_window or {}
        self.fail_windows = set(fail_windows)
        self.calls = 0
        self.systems = []
        self.last_usage = None

    def complete_json(self, system, prompt, schema, examples=()):
        self.calls += 1
        self.systems.append(system)
        targets = schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"]
        key = tuple(targets)
        if key in self.fail_windows:
            raise AtomizationRequestError("scripted failure")
        self.last_usage = {"prompt_eval_count": 100, "eval_count": 10, "done_reason": "stop"}
        if key in self.raw_by_window:
            return self.raw_by_window[key]
        out = []
        for mid in targets:
            claims, relations = self.entries.get(mid, ([], []))
            out.append({"source_message_id": mid, "status": "claims" if claims else "no_durable_assertion",
                        "claims": claims, "explicit_relations": relations})
        return json.dumps({"messages": out})


def mem(cid, content, memory_type="personal_fact", entities=(), temporal="", polarity="positive"):
    return {"claim_id": cid, "content": content, "memory_type": memory_type, "entities": list(entities),
            "qualifiers": [], "temporal": {"expression": temporal, "resolved_date": ""}, "polarity": polarity,
            "source_message_id": cid.split(":c")[0]}


def conv(rows):
    return Conversation(CID, tuple(Message(mid, CID, i, role, text, ts) for i, (mid, role, text, ts) in enumerate(rows)))


def run(tmp_path, conversation, client, name="m.db"):
    store = Store(str(tmp_path / name))
    report = ingest_imported(conversation, store, extract="memories", atomizer=SemanticAtomizer(client, MEMORY_PROTOCOL))
    semantic = SemanticStore(store.connection)
    claims = semantic.get_claims(CID)
    relations = semantic.get_relations(CID)
    records = memory_records(conversation, claims, relations)
    return report, claims, relations, records, store


TS = "2023-05-01T10:00:00"


def test_one_chunk_yields_multiple_independent_typed_memories(tmp_path):
    c = conv([("m1", "user", "I'm Priya. I work as a nurse and I hate early shifts.", TS)])
    client = ScriptedClient({"m1": ([mem("m1:c1", "The user is Priya.", entities=["Priya"]),
                                     mem("m1:c2", "Priya works as a nurse.", entities=["Priya"]),
                                     mem("m1:c3", "Priya hates early shifts.", "preference", ["Priya"])], [])})
    report, claims, _, records, _ = run(tmp_path, c, client)
    assert client.systems == [MEMORY_SYSTEM_PROMPT]
    assert [r.content for r in records] == ["The user is Priya.", "Priya works as a nurse.", "Priya hates early shifts."]
    assert [r.memory_type for r in records] == ["personal_fact", "personal_fact", "preference"]
    assert all(c.claim_type is None for c in claims)  # memory type never becomes a consolidation slot
    assert {r.status for r in records} == {"current"}
    assert report.model_calls == 1 and report.prompt_tokens == 100 and report.output_tokens == 10


def test_chunk_without_durable_information_yields_no_memories(tmp_path):
    c = conv([("m1", "user", "Hi! How are you?", TS), ("m2", "assistant", "Great, thanks! What's up?", TS)])
    report, claims, _, records, _ = run(tmp_path, c, ScriptedClient())
    assert claims == [] and records == []
    assert report.statuses["no_durable_assertion"] == 2 and report.failed_windows == 0


def test_two_different_preferences_do_not_supersede_each_other(tmp_path):
    c = conv([("m1", "user", "I love jazz.", TS), ("m2", "user", "I also love hiking.", TS)])
    client = ScriptedClient({"m1": ([mem("m1:c1", "The user loves jazz.", "preference")], []),
                             "m2": ([mem("m2:c1", "The user also loves hiking.", "preference")], [])})
    _, _, _, records, _ = run(tmp_path, c, client)
    assert [r.status for r in records] == ["current", "current"]


def test_repeated_fact_across_chunks_is_one_memory_with_both_sources(tmp_path):
    rows = [("m1", "user", "I'm allergic to penicillin.", TS)] + [
        (f"f{i}", "user", f"filler {i}", TS) for i in range(4)] + [("m6", "user", "I'm allergic to penicillin.", "2023-06-01T10:00:00")]
    client = ScriptedClient({"m1": ([mem("m1:c1", "The user is allergic to penicillin.")], []),
                             "m6": ([mem("m6:c1", "The user is allergic to penicillin.")], [])})
    report, claims, _, records, _ = run(tmp_path, conv(rows), client)
    assert report.windows == 2 and len(claims) == 2
    assert len(records) == 1
    assert [s.message_id for s in records[0].sources] == ["m1", "m6"]
    assert [s.timestamp for s in records[0].sources] == [TS, "2023-06-01T10:00:00"]


def test_explicit_update_supersedes_but_an_unlinked_change_keeps_both(tmp_path):
    rows = [("m1", "user", "I live in Porto.", TS),
            ("m2", "user", "Actually I moved to Lisbon, I don't live in Porto anymore.", "2023-07-01T10:00:00"),
            ("m3", "user", "I live in Braga.", "2023-08-01T10:00:00")]
    client = ScriptedClient({
        "m1": ([mem("m1:c1", "The user lives in Porto.", entities=["Porto"])], []),
        "m2": ([mem("m2:c1", "The user moved to Lisbon.", "event", ["Lisbon"])],
               [{"from_claim_id": "m2:c1", "to_claim_id": "m1:c1", "type": "UPDATES", "source_message_id": "m2",
                 "evidence": "Actually I moved to Lisbon"}]),
        "m3": ([mem("m3:c1", "The user lives in Braga.", entities=["Braga"])], []),
    })
    _, _, relations, records, _ = run(tmp_path, conv(rows), client)
    by = {r.content: r for r in records}
    assert [r.type for r in relations] == ["UPDATES"]
    assert by["The user lives in Porto."].status == "superseded"
    assert by["The user lives in Porto."].superseded_by == by["The user moved to Lisbon."].memory_id
    assert by["The user moved to Lisbon."].status == "current"
    # No relation and no reliable slot: the Braga statement is kept, nothing is silently superseded by it.
    assert by["The user lives in Braga."].status == "current"
    assert by["The user moved to Lisbon."].superseded_by is None
    assert by["The user lives in Porto."].sources[0].message_id == "m1"  # historical evidence kept


def test_past_state_is_historical_not_current(tmp_path):
    c = conv([("m1", "user", "I used to live in Delhi. Now I live in Pune.", TS)])
    client = ScriptedClient({"m1": ([mem("m1:c1", "The user used to live in Delhi.", entities=["Delhi"]),
                                     mem("m1:c2", "Now the user lives in Pune.", entities=["Pune"])], [])})
    _, _, _, records, _ = run(tmp_path, c, client)
    by = {r.content: r.status for r in records}
    assert by["The user used to live in Delhi."] == "historical"
    assert by["Now the user lives in Pune."] == "current"


def test_speaker_attribution_and_provenance(tmp_path):
    rows = [("d1", "user", "Priya: I paint landscapes every weekend.", TS),
            ("d2", "assistant", "Lena: You should try watercolours!", TS),
            ("d3", "assistant", "Lena: I booked us a pottery class for Friday.", TS)]
    client = ScriptedClient({
        "d1": ([mem("d1:c1", "Priya paints landscapes every weekend.", "preference", ["Priya"])], []),
        "d3": ([mem("d3:c1", "Lena booked a pottery class for Friday.", "event", ["Lena"], temporal="for Friday")], []),
    })
    _, _, _, records, _ = run(tmp_path, conv(rows), client)
    by = {r.content: r for r in records}
    src = by["Priya paints landscapes every weekend."].sources[0]
    assert (src.message_id, src.speaker, src.role, src.timestamp) == ("d1", "Priya", "user", TS)
    assert by["Lena booked a pottery class for Friday."].sources[0].speaker == "Lena"
    assert not any("watercolour" in r.content for r in records)  # a suggestion is not a fact about anyone
    assert all(r.assertion == "explicit" for r in records)


def test_malformed_and_invalid_output_is_rejected_safely(tmp_path):
    rows = [("a1", "user", "I like cats.", TS)] + [(f"b{i}", "user", f"filler {i}", TS) for i in range(3)] + [
        ("c1", "user", "My dog is called Rex.", TS)] + [(f"d{i}", "user", f"filler {i}", TS) for i in range(3)] + [
        ("e1", "user", "I play chess.", TS)]
    client = ScriptedClient(
        entries={"c1": ([mem("c1:c1", "The user's dog is called Rex.", "pet_fact", ["Rex"]),  # unknown type
                         mem("c1:c2", "The user lives in Paris.")], [])},  # not stated by the source
        raw_by_window={("a1", "b0", "b1", "b2"): "{not json"},
        fail_windows={("e1",)},
    )
    report, claims, _, records, _ = run(tmp_path, conv(rows), client)
    assert report.windows == 3 and report.failed_windows == 2
    assert report.issues["INVALID_JSON:rejected"] == 1
    assert report.issues["INVALID_MEMORY_TYPE:flagged"] == 1 and report.issues["UNSUPPORTED_CLAIM:rejected"] == 1
    assert [c.content for c in claims] == ["The user's dog is called Rex."]
    assert records[0].memory_type is None


def test_reprocessing_the_same_source_creates_no_duplicates(tmp_path):
    c = conv([("m1", "user", "I'm Priya and I hate early shifts.", TS)])
    entries = {"m1": ([mem("m1:c1", "The user is Priya.", entities=["Priya"]),
                       mem("m1:c2", "Priya hates early shifts.", "preference", ["Priya"])], [])}
    store = Store(str(tmp_path / "same.db"))
    first = second = None
    for _ in range(2):
        ingest_imported(c, store, extract="memories", atomizer=SemanticAtomizer(ScriptedClient(entries), MEMORY_PROTOCOL))
        semantic = SemanticStore(store.connection)
        recs = memory_records(c, semantic.get_claims(CID), semantic.get_relations(CID))
        first, second = second, [r.memory_id for r in recs]
    assert first == second == ["mem:m1:c1", "mem:m1:c2"]
    assert len(SemanticStore(store.connection).get_claims(CID)) == 2


def test_unknown_extract_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ingest_imported(conv([("m1", "user", "hi", TS)]), Store(str(tmp_path / "x.db")), extract="summaries")


def test_extracted_memories_work_with_graph_hybrid_retrieval(tmp_path, monkeypatch):
    from continuator_memory.webapi import locomo_report as lr
    from continuator_memory.webapi.workspace import Workspace

    monkeypatch.setenv("CONTINUATOR_LEXICAL_WEIGHT", "0.25")
    monkeypatch.setenv("CONTINUATOR_CONTEXT_WEIGHT", "0.5")
    rows = [("d1", "user", "Priya: I paint landscapes every weekend.", TS),
            ("d2", "assistant", "Lena: Lovely! I started a pottery class in Brooklyn.", TS),
            ("d3", "user", "Priya: My favourite colour is blue.", TS)]
    client = ScriptedClient({
        "d1": ([mem("d1:c1", "Priya paints landscapes every weekend.", "preference", ["Priya"])], []),
        "d2": ([mem("d2:c1", "Lena started a pottery class in Brooklyn.", "event", ["Lena", "Brooklyn"])], []),
        "d3": ([mem("d3:c1", "Priya's favourite colour is blue.", "preference", ["Priya"])], []),
    })
    _, _, _, _, store = run(tmp_path, conv(rows), client)
    store.close()
    ws = Workspace.from_sqlite(tmp_path / "m.db", embedder=HashingEmbedder(), k=5)
    ws._eval_retriever = lr.build_memory_retriever(ws.conversation, ws.claims, ws.relations, ws.embedder(),
                                                    lr.retrieval_config("hybrid"))
    gold = ["d2"]
    item = lr.LocomoQuestion(0, "Where is Lena's pottery class?", "Brooklyn", 1, gold,
                             lr.classify_evidence(gold, {m.message_id for m in ws.conversation.messages}))
    row = lr.evaluate_question(item, ws, mode="graph_hybrid", top_k=5, depth=10, generate=False)
    assert row["error"] is None
    assert row["promptMessageIds"] and set(row["promptMessageIds"]) <= {"d1", "d2", "d3"}  # original turns
    assert row["metrics"]["goldRanks"]["d2"] is not None
    assert {m["memoryId"] for m in row["retrievedMemories"]} <= {"mem:d1:c1", "mem:d2:c1", "mem:d3:c1"}
