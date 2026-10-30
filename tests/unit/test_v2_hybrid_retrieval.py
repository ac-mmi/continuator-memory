import ast
import inspect

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.pipeline import Pipeline
from continuator_memory.v2 import hybrid_retrieval
from continuator_memory.v2.hybrid_retrieval import (
    DEFAULT_RRF_C,
    EVIDENCE_MEMORY_SOURCE,
    EVIDENCE_RAW,
    EVIDENCE_RAW_AND_MEMORY,
    HybridConfig,
    HybridRetriever,
)
from continuator_memory.v2.memory import (
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    Memory,
    MemoryLink,
    MemorySource,
)
from continuator_memory.v2.memory_retrieval import EXCLUDED_SUPERSEDED, MemoryRetriever
from tests.helpers import HashingEmbedder

CONV = "conv"
TIME_CLASS = {STATUS_CURRENT: "present", STATUS_SUPERSEDED: "present", STATUS_HISTORICAL: "past",
              STATUS_CONTESTED: "present", STATUS_FUTURE: "future"}


def conversation(*texts, cid=CONV):
    msgs = []
    for i, t in enumerate(texts, 1):
        role, content = t if isinstance(t, tuple) else ("user", t)
        msgs.append(Message(f"m{i}", cid, i - 1, role, content, f"2026-01-{i:02d}T10:00:00"))
    return Conversation(cid, tuple(msgs))


def memory(n, content, status=STATUS_CURRENT, message=None, superseded_by=None, cid=CONV):
    message = message or f"m{n}"
    return Memory(f"mem:{message}:c{n}", cid, content, status, f"{message}:c{n}", message, n,
                  superseded_by=superseded_by, time_class=TIME_CLASS[status])


def src(mem, message, claim=None):
    return MemorySource(mem.memory_id, claim or f"{message}:c9", message, int(message[1:]), None)


def build(store, conv, memories, sources=None, links=(), config=None, window=1, overlap=0):
    embedder = HashingEmbedder(dim=4096)
    pipeline = Pipeline(store, embedder, window_size=window, overlap=overlap)
    pipeline.ingest(conv)
    mr = MemoryRetriever(embedder)
    if sources is None:
        sources = [MemorySource(m.memory_id, m.claim_id, m.message_id, m.position, None) for m in memories]
    mr.index(memories, sources, links, conversation_id=conv.conversation_id)
    return HybridRetriever(pipeline.retriever, mr, store, config)


def by_message(results):
    return {r.message_id: r for r in results}


def mids(results):
    return [r.message_id for r in results]


FILLER = ("I do yoga on Sundays.", "I do enjoy long walks.", "Where do you buy coffee?", "I do like pasta.")


# ---- A–L fixtures -----------------------------------------------------------------------


def test_a_raw_only_evidence_survives(store):
    conv = conversation("The ferry to the island leaves at seven because the tide changes early.",
                        "I like green tea.")
    h = build(store, conv, [memory(2, "I like green tea.")])
    [first, *_] = h.retrieve("When does the ferry leave?")
    assert first.message_id == "m1" and first.sources == ("raw",) and first.evidence_type == EVIDENCE_RAW
    assert first.text.startswith("The ferry") and first.memories == () and first.raw_rank == 1


def test_b_memory_only_source_appears(store):
    conv = conversation("We finally signed the lease near Indiranagar last spring.", *FILLER)
    lease = memory(1, "I live in Bangalore.")
    h = build(store, conv, [lease], config=HybridConfig(raw_k=2))
    outcome = h.search("Where do I live?")
    assert "m1" not in {mid for c in outcome.raw for mid in c.source_message_ids}
    m1 = by_message(outcome.results)["m1"]
    assert m1.sources == ("memory",) and m1.evidence_type == EVIDENCE_MEMORY_SOURCE
    assert m1.text == "We finally signed the lease near Indiranagar last spring."
    assert (m1.role, m1.timestamp, m1.position) == ("user", "2026-01-01T10:00:00", 0)
    assert m1.memory_ids == (lease.memory_id,) and m1.claim_ids == (lease.claim_id,)
    assert m1.memories[0].content == "I live in Bangalore." and m1.memories[0].status == STATUS_CURRENT
    assert m1.raw_rank is None and m1.raw_score is None


def test_c_both_paths_find_same_message_once(store):
    conv = conversation("I live in Bangalore.", "The weather is mild.")
    m = memory(1, "I live in Bangalore.")
    results = build(store, conv, [m]).retrieve("Where do I live?")
    assert mids(results).count("m1") == 1
    r = by_message(results)["m1"]
    assert r.sources == ("raw", "memory") and r.evidence_type == EVIDENCE_RAW_AND_MEMORY
    assert r.raw_rank == 1 and r.memory_rank == 1 and r.raw_score > 0 and r.memory_similarity > 0
    assert r.fusion_score == pytest.approx(2 / (DEFAULT_RRF_C + 1))
    assert results[0].message_id == "m1"


def test_d_memory_with_multiple_sources(store):
    conv = conversation("I live in Delhi.", "Lunch was great.", "Still in Delhi, as I said.", "Rain again.")
    m = memory(1, "I live in Delhi.")
    sources = [src(m, "m1", "m1:c1"), src(m, "m3", "m3:c1"), src(m, "m3", "m3:c2")]
    outcome = build(store, conv, [m], sources=sources, config=HybridConfig(final_k=10)).search("Where do I live?")
    assert [r.memory_id for r in outcome.memory.results] == [m.memory_id]
    with_memory = [r for r in outcome.results if r.memories]
    assert sorted(mids(with_memory)) == ["m1", "m3"]
    assert len(mids(outcome.results)) == len(set(mids(outcome.results)))
    assert by_message(with_memory)["m3"].claim_ids == ("m3:c1", "m3:c2")
    assert by_message(with_memory)["m1"].claim_ids == ("m1:c1",)


def test_e_raw_ranking_preserved_for_raw_only_messages(store):
    conv = conversation("Our flight to Lisbon departs Friday.", "The Lisbon flight has a stop in Madrid.",
                        "I bought new shoes.", "The flight to Lisbon is cheap.")
    h = build(store, conv, [], config=HybridConfig(raw_k=4, final_k=4))
    outcome = h.search("When is the flight to Lisbon?")
    raw_order = [c.source_message_ids[0] for c in outcome.raw]
    assert mids(outcome.results) == raw_order
    assert [r.raw_rank for r in outcome.results] == [1, 2, 3, 4]


def test_f_superseded_memory_not_reintroduced_via_memory_path(store):
    conv = conversation("Back then my flat was in Delhi.", "I live in Bangalore now.", *FILLER)
    delhi = memory(1, "I live in Delhi.", STATUS_SUPERSEDED, superseded_by="mem:m2:c2")
    bangalore = memory(2, "I live in Bangalore.")
    links = [MemoryLink(bangalore.memory_id, delhi.memory_id, LINK_SUPERSEDES, bangalore.claim_id)]
    outcome = build(store, conv, [delhi, bangalore], links=links, config=HybridConfig(raw_k=2)).search(
        "Where do I live?")
    assert (delhi.memory_id, EXCLUDED_SUPERSEDED) in outcome.memory.excluded
    assert "m1" not in mids(outcome.results)
    assert all(delhi.memory_id not in r.memory_ids for r in outcome.results)


def test_f_superseded_source_may_stay_as_raw_evidence(store):
    conv = conversation("I live in Delhi.", "Moved! Bangalore is home now.")
    delhi = memory(1, "I live in Delhi.", STATUS_SUPERSEDED, superseded_by="mem:m2:c2")
    bangalore = memory(2, "I live in Bangalore.")
    links = [MemoryLink(bangalore.memory_id, delhi.memory_id, LINK_SUPERSEDES, bangalore.claim_id)]
    results = build(store, conv, [delhi, bangalore], links=links).retrieve("Where do I live?")
    m1 = by_message(results)["m1"]
    assert m1.sources == ("raw",) and m1.memories == ()
    assert by_message(results)["m2"].memory_ids == (bangalore.memory_id,)


def test_g_historical_question_exposes_superseded_source(store):
    conv = conversation("Back then my flat was in Delhi.", "I live in Bangalore now.", *FILLER)
    delhi = memory(1, "I lived in Delhi before.", STATUS_SUPERSEDED, superseded_by="mem:m2:c2")
    bangalore = memory(2, "I live in Bangalore.")
    links = [MemoryLink(bangalore.memory_id, delhi.memory_id, LINK_SUPERSEDES, bangalore.claim_id)]
    results = build(store, conv, [delhi, bangalore], links=links,
                    config=HybridConfig(raw_k=2)).retrieve("Where did I live before?")
    m1 = by_message(results)["m1"]
    assert m1.text == "Back then my flat was in Delhi." and "memory" in m1.sources
    assert m1.memories[0].status == STATUS_SUPERSEDED
    assert [(l.type, l.role, l.memory_id) for l in m1.memories[0].links] == [
        (LINK_SUPERSEDES, "newer", bangalore.memory_id)]


def test_g_historical_memory_source_available(store):
    conv = conversation("As a kid I was obsessed with chess.", *FILLER)
    chess = memory(1, "I used to play chess.", STATUS_HISTORICAL)
    results = build(store, conv, [chess], config=HybridConfig(raw_k=2)).retrieve("What did I used to play?")
    m1 = by_message(results)["m1"]
    assert "memory" in m1.sources and m1.memories[0].status == STATUS_HISTORICAL
    assert m1.memories[0].time_class == "past"


def test_h_future_memory_source_available(store):
    conv = conversation("The relocation paperwork for Mumbai should clear by spring.", *FILLER)
    mumbai = memory(1, "I will move to Mumbai next year.", STATUS_FUTURE)
    results = build(store, conv, [mumbai], config=HybridConfig(raw_k=2)).retrieve("Where am I planning to move?")
    m1 = by_message(results)["m1"]
    assert "memory" in m1.sources and m1.memories[0].status == STATUS_FUTURE
    assert m1.text.startswith("The relocation paperwork")


def test_i_contested_memories_both_survive(store):
    conv = conversation("Project launch is July 1.", "No wait, project launch is July 15.", "Lunch?")
    a = memory(1, "Project launch is July 1.", STATUS_CONTESTED)
    b = memory(2, "Project launch is July 15.", STATUS_CONTESTED)
    links = [MemoryLink(b.memory_id, a.memory_id, LINK_CONTRADICTS, b.claim_id)]
    results = build(store, conv, [a, b], links=links).retrieve("When is the project launch?")
    found = by_message(results)
    assert {"m1", "m2"} <= set(found)
    assert found["m1"].memories[0].status == found["m2"].memories[0].status == STATUS_CONTESTED
    assert [l.type for l in found["m1"].memories[0].links] == [LINK_CONTRADICTS]


def test_j_raw_memory_conflict_both_available(store):
    conv = conversation("I live in Delhi.", "Packing boxes all week.", "Settled in nicely after the move.")
    bangalore = memory(3, "I live in Bangalore.")
    results = build(store, conv, [bangalore]).retrieve("Where do I live?")
    found = by_message(results)
    assert found["m1"].text == "I live in Delhi." and found["m1"].sources == ("raw",)
    assert found["m3"].memories[0].content == "I live in Bangalore." and "memory" in found["m3"].sources


def test_k_irrelevant_memories_do_not_flood(store):
    conv = conversation("I live in Bangalore.", "My cat is named Miso.", "The dog sleeps all day.",
                        "Green is my favourite colour.", "The car needs new tyres.", "Rain again.")
    relevant = memory(1, "I live in Bangalore.")
    noise = [memory(i, text) for i, text in ((2, "My cat is named Miso."), (3, "The dog sleeps all day."),
                                             (4, "Green is my favourite colour."), (5, "The car needs new tyres."))]
    outcome = build(store, conv, [relevant, *noise]).search("Where do I live?")
    assert outcome.results[0].message_id == "m1"
    assert [r.memory_id for r in outcome.memory.results] == [relevant.memory_id]
    assert all(r.memory_ids in ((), (relevant.memory_id,)) for r in outcome.results)


def test_l_equal_fusion_scores_have_stable_order(store):
    conv = conversation("Where do I buy coffee?", "We finally signed the lease near Indiranagar.", "Rain again.")
    lease = memory(2, "I live in Bangalore.")
    h = build(store, conv, [lease], config=HybridConfig(raw_k=1))
    first = h.search("Where do I live?")
    assert [(r.message_id, r.raw_rank, r.memory_rank) for r in first.results] == [("m1", 1, None), ("m2", None, 1)]
    assert first.results[0].fusion_score == first.results[1].fusion_score
    for _ in range(3):
        again = h.search("Where do I live?")
        assert [r.to_dict() for r in again.results] == [r.to_dict() for r in first.results]
        assert again.debug() == first.debug()


# ---- chunk granularity ------------------------------------------------------------------


def test_window_chunks_expand_to_messages_without_duplicates(store):
    conv = conversation("I live in Bangalore.", ("assistant", "Nice city."), "The traffic is bad.",
                        ("assistant", "Try the metro."), "My cat is Miso.", ("assistant", "Cute."))
    h = build(store, conv, [memory(1, "I live in Bangalore.")], window=4, overlap=2,
              config=HybridConfig(raw_k=3, final_k=10))
    outcome = h.search("Where do I live?")
    assert len(mids(outcome.results)) == len(set(mids(outcome.results))) == outcome.candidates
    first = outcome.results[0]
    assert first.message_id == "m1" and first.sources == ("raw", "memory")
    raw_only = [r for r in outcome.results if r.sources == ("raw",)]
    assert all(r.raw_rank >= first.raw_rank for r in raw_only)
    multi = [r for r in outcome.results if len(r.chunk_ids) > 1]
    assert multi and all(r.raw_score == max(c.score for c in outcome.raw if r.message_id in c.source_message_ids)
                         for r in multi)
    assert {r.role for r in outcome.results} == {"user", "assistant"}


# ---- source expansion safety ------------------------------------------------------------


def test_expansion_uses_only_direct_sources_of_retrieved_memories(store):
    conv = conversation("Filler one.", "Filler two.", "My sister Leah works at a bakery.", "Filler four.",
                        "Leah lives in Valencia.", "Filler six.")
    bakery = memory(3, "Leah works at a bakery.")
    home = memory(5, "Leah lives in Valencia.")
    links = [MemoryLink(home.memory_id, bakery.memory_id, LINK_REFERS_TO, home.claim_id)]
    outcome = build(store, conv, [bakery, home], links=links,
                    config=HybridConfig(raw_k=1, memory_k=1, final_k=10)).search("Where does Leah live in Valencia?")
    assert [r.memory_id for r in outcome.memory.results] == [home.memory_id]
    assert outcome.candidates == len(outcome.results)
    assert set(mids(outcome.results)) == {"m5"}


def test_missing_source_message_is_not_fabricated(store):
    conv = conversation("I like tea.")
    ghost = memory(9, "I live in Bangalore.", message="m9")
    results = build(store, conv, [ghost]).retrieve("Where do I live?")
    r = by_message(results)["m9"]
    assert r.text is None and r.role is None and r.sources == ("memory",) and r.memory_ids == (ghost.memory_id,)


def test_conversation_scope_is_respected(store):
    embedder = HashingEmbedder(dim=4096)
    pipeline = Pipeline(store, embedder, window_size=1, overlap=0)
    pipeline.ingest(conversation("I live in Delhi.", cid="a"))
    pipeline.ingest(conversation("I live in Bangalore.", cid="b"))
    mr = MemoryRetriever(embedder)
    for cid, text in (("a", "I live in Delhi."), ("b", "I live in Bangalore.")):
        m = memory(1, text, cid=cid)
        mr.index([m], [MemorySource(m.memory_id, m.claim_id, "m1", 0, None)])
    h = HybridRetriever(pipeline.retriever, mr, store)
    assert {r.conversation_id for r in h.retrieve("Where do I live?", conversation_id="b")} == {"b"}
    both = h.retrieve("Where do I live?")
    assert sorted((r.conversation_id, r.message_id) for r in both) == [("a", "m1"), ("b", "m1")]
    assert all(r.sources == ("raw", "memory") for r in both)


# ---- K semantics and delegation ---------------------------------------------------------


class Spy:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def retrieve(self, question, k, conversation_id=None):
        self.calls.append(("retrieve", k, conversation_id))
        return self.inner.retrieve(question, k, conversation_id)

    def search(self, question, k=None, conversation_id=None):
        self.calls.append(("search", k, conversation_id))
        return self.inner.search(question, k, conversation_id)


def test_k_limits_are_separate_and_bounded(store):
    conv = conversation(*(f"I visited city number {i} on a trip." for i in range(1, 13)))
    memories = [memory(i, f"I visited city number {i} on a trip.") for i in range(1, 13)]
    h = build(store, conv, memories, config=HybridConfig(raw_k=3, memory_k=2, final_k=4))
    raw_spy, mem_spy = Spy(h.raw_retriever), Spy(h.memory_retriever)
    h.raw_retriever, h.memory_retriever = raw_spy, mem_spy
    outcome = h.search("Which city did I visit on a trip?", conversation_id=CONV)
    assert raw_spy.calls == [("retrieve", 3, CONV)] and mem_spy.calls == [("search", 2, CONV)]
    assert len(outcome.raw) == 3 and len(outcome.memory.results) == 2
    assert outcome.candidates <= 3 + 2
    assert len(outcome.results) == min(4, outcome.candidates)
    assert [r.rank for r in outcome.results] == list(range(1, len(outcome.results) + 1))
    wide = HybridRetriever(raw_spy.inner, mem_spy.inner, store, HybridConfig(raw_k=10, memory_k=10, final_k=4))
    assert len(wide.retrieve("Which city did I visit on a trip?")) == 4
    assert len(h.retrieve("Which city did I visit on a trip?", k=1)) == 1
    with pytest.raises(ValueError):
        h.retrieve("x", k=0)
    with pytest.raises(ValueError):
        HybridConfig(raw_k=0)


def test_defaults_are_five():
    config = HybridConfig()
    assert (config.raw_k, config.memory_k, config.final_k, config.rrf_c) == (5, 5, 5, 60)


def test_phase_3a_output_is_used_unchanged(store):
    conv = conversation("I live in Bangalore.", "I lived in Delhi.")
    ms = [memory(1, "I live in Bangalore."), memory(2, "I lived in Delhi.", STATUS_HISTORICAL)]
    h = build(store, conv, ms)
    outcome = h.search("Where do I live?")
    assert outcome.memory == h.memory_retriever.search("Where do I live?", 5)
    assert list(outcome.raw) == h.raw_retriever.retrieve("Where do I live?", 5)


# ---- diagnostics ------------------------------------------------------------------------


def test_debug_representation(store):
    conv = conversation("I live in Bangalore.", "We finally signed the lease near Indiranagar.", *FILLER)
    m = memory(2, "I live in Bangalore.")
    outcome = build(store, conv, [m], config=HybridConfig(raw_k=2)).search("Where do I live?")
    rows = outcome.debug()["results"]
    assert set(rows[0]) == {"rank", "message_id", "fusion_score", "raw_rank", "raw_score", "memory_rank",
                            "memory_score", "sources", "memory_ids", "status"}
    row = next(r for r in rows if r["message_id"] == "m2")
    assert row["sources"] == ["memory"] and row["memory_ids"] == [m.memory_id] and row["status"] == ["current"]
    assert row["raw_rank"] is None and row["memory_rank"] == 1
    d = outcome.results[0].to_dict()
    for key in ("message_id", "conversation_id", "text", "role", "timestamp", "memories", "sources",
                "evidence_type", "fusion_score", "raw_score", "memory_similarity", "chunk_ids"):
        assert key in d


def test_module_does_no_generation_or_new_search():
    tree = ast.parse(inspect.getsource(hybrid_retrieval))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m and ("generation" in m or "ollama" in m or "embedding" in m) for m in imported)
    text = inspect.getsource(hybrid_retrieval)
    assert ".embed(" not in text and "cosine_similarity" not in text
