from continuator_memory.models import parse_conversation
from continuator_memory.v2.consolidation import Consolidator
from continuator_memory.v2.memory import STATUS_CURRENT, STATUS_SUPERSEDED, UPDATE
from continuator_memory.v2.memory_store import MemoryStore, consolidate_stored
from tests.unit.test_v2_pipeline import CONVERSATION, pipeline


def _ingest(store):
    p, _ = pipeline(store)
    p.ingest(parse_conversation(CONVERSATION))
    return p


def test_consolidate_stored_claims_and_round_trip(store):
    _ingest(store)
    c = consolidate_stored(store.connection, "trip")
    ms = MemoryStore(store.connection)
    memories = ms.get_memories("trip")
    assert [m.to_dict() for m in memories] == [m.to_dict() for m in c._ordered()]
    by_claim = {m.claim_id: m for m in memories}
    assert by_claim["m1:c2"].status == STATUS_SUPERSEDED and by_claim["m1:c2"].superseded_by == "mem:m4:c1"
    assert by_claim["m4:c1"].status == STATUS_CURRENT
    assert [(l.newer_memory_id, l.older_memory_id, l.type, l.relation_id) for l in ms.get_links("trip")] == [
        ("mem:m4:c1", "mem:m1:c2", "UPDATES", "m4:r1")]
    decisions = ms.get_decisions("trip")
    assert [d.to_dict() for d in decisions] == [d.to_dict() for d in c.decisions]
    assert next(d for d in decisions if d.claim_id == "m4:c1").action == UPDATE
    assert "Leah lives in Madrid." not in [m.content for m in ms.get_memories("trip", current_only=True)]


def test_superseded_memory_source_stays_queryable_and_traces_to_raw_message(store):
    _ingest(store)
    consolidate_stored(store.connection, "trip")
    ms = MemoryStore(store.connection)
    [source] = ms.get_sources("trip", "mem:m1:c2")
    assert (source.claim_id, source.message_id, source.observed_at) == ("m1:c2", "m1", "2026-02-01T09:00:00")
    raw = store.get_messages("trip", [source.message_id])[0]
    assert raw.content == "My sister Leah lives in Madrid."
    columns = {row[1] for row in store.connection.execute("PRAGMA table_info(v2_memory_sources)")}
    assert "content" not in columns


def test_save_replaces_snapshot_and_is_deterministic(store):
    _ingest(store)
    consolidate_stored(store.connection, "trip")
    ms = MemoryStore(store.connection)
    first = ([m.to_dict() for m in ms.get_memories("trip")], ms.get_sources("trip"), ms.get_links("trip"))
    consolidate_stored(store.connection, "trip")
    second = ([m.to_dict() for m in ms.get_memories("trip")], ms.get_sources("trip"), ms.get_links("trip"))
    assert first == second
    ms.save(Consolidator("trip"))
    assert ms.get_memories("trip") == [] and ms.get_decisions("trip") == []


def test_memory_tables_are_additive(store):
    before = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    MemoryStore(store.connection)
    after = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert after - before == {"v2_memories", "v2_memory_sources", "v2_memory_versions", "v2_memory_decisions"}
    assert before <= after
