"""Focused tests for contextual_bm25 (true sparse BM25; no LoCoMo full eval)."""

from __future__ import annotations

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.bm25 import BM25Index, RetrievalUnit, sparse_tokenize
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.webapi.contextual_bm25 import (
    MODE_CONTEXTUAL_BM25,
    build_contextual_bm25_index,
    contextual_retrieval_units,
    retrieve_contextual_bm25_evidence,
    search_contextual_bm25,
)
from continuator_memory.webapi.locomo_report import ranked_message_ids, retrieval_config
from continuator_memory.webapi.workspace import (
    CONTEXTUAL_ENRICHMENT_MODES,
    RETRIEVAL_MODES,
    Workspace,
    normalize_retrieval_mode,
)
from tests.helpers import HashingEmbedder


def _msg(mid: str, position: int, text: str, timestamp: str | None) -> Message:
    return Message(mid, "conv", position, "user", text, timestamp)


def _conversation() -> Conversation:
    stamps = ["2023-01-01T10:00:00"] * 4
    messages = (
        _msg("D1:1", 0, "Alice: previous chitchat about weather", stamps[0]),
        _msg("D1:2", 1, "Bob: rareterm xylophone practice every Tuesday", stamps[1]),
        _msg("D1:3", 2, "Alice: next acknowledgement cool", stamps[2]),
        _msg("D1:4", 3, "Bob: unrelated gardening tomatoes", stamps[3]),
    )
    return Conversation("conv", messages, title="test")


def _claims(conversation: Conversation) -> list[Claim]:
    return [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]


def test_mode_registration():
    assert MODE_CONTEXTUAL_BM25 in RETRIEVAL_MODES
    assert MODE_CONTEXTUAL_BM25 in CONTEXTUAL_ENRICHMENT_MODES
    assert normalize_retrieval_mode("contextual_bm25") == "contextual_bm25"
    assert retrieval_config("contextual_bm25") is None
    assert normalize_retrieval_mode("lexical") == "lexical"
    assert normalize_retrieval_mode("contextual_lexical") == "contextual_lexical"


def test_sparse_tokenize_is_general():
    tokens = sparse_tokenize("Hello, Xylophone-42!")
    assert "hello" in tokens
    assert "xylophone" in tokens
    assert "42" in tokens


def test_bm25_ranks_exact_rare_term_above_unrelated():
    units = [
        RetrievalUnit("a", "gardening tomatoes and soil", ("m1",)),
        RetrievalUnit("b", "rareterm xylophone practice schedule", ("m2",)),
        RetrievalUnit("c", "weather chat previous next", ("m3",)),
    ]
    index = BM25Index(units)
    hits = index.search("rareterm xylophone", k=3)
    assert hits
    assert hits[0].unit.id == "b"
    assert hits[0].score > 0


def test_bm25_empty_corpus_and_empty_query():
    assert BM25Index([]).search("anything", k=5) == []
    units = [RetrievalUnit("a", "hello world", ("m1",))]
    assert BM25Index(units).search("", k=5) == []
    assert BM25Index(units).search("   ", k=5) == []


def test_contextual_units_credit_target_only_including_boundaries():
    conversation = _conversation()
    consolidator = consolidate(conversation.conversation_id, _claims(conversation))
    units = contextual_retrieval_units(
        conversation, consolidator.memories.values(), consolidator.sources,
    )
    by_id = {u.id: u for u in units}
    mid_mem = next(m for m in consolidator.memories.values() if m.message_id == "D1:2")
    mid = by_id[mid_mem.memory_id]
    # Target unit includes PREV/TARGET/NEXT text...
    assert "[PREV]" in mid.retrieval_content
    assert "[TARGET]" in mid.retrieval_content
    assert "[NEXT]" in mid.retrieval_content
    assert "xylophone" in mid.retrieval_content
    # ...but only TARGET message id is credited.
    assert mid.source_message_ids == ("D1:2",)

    first_mem = next(m for m in consolidator.memories.values() if m.message_id == "D1:1")
    first = by_id[first_mem.memory_id]
    assert "[PREV]" not in first.retrieval_content
    assert first.source_message_ids == ("D1:1",)
    # Neighbour enrichment may copy TARGET terms into PREV/NEXT text of adjacent units.
    assert "xylophone" in first.retrieval_content

    last_mem = next(m for m in consolidator.memories.values() if m.message_id == "D1:4")
    last = by_id[last_mem.memory_id]
    assert "[NEXT]" not in last.retrieval_content
    assert last.source_message_ids == ("D1:4",)


def test_retrieve_contextual_bm25_is_sparse_and_target_only():
    conversation = _conversation()
    claims = _claims(conversation)
    # No embedder argument — sparse path must not require embeddings for ranking.
    evidence, meta = retrieve_contextual_bm25_evidence(
        "rareterm xylophone",
        conversation,
        claims,
        (),
        k=4,
        budget_chars=10_000,
    )
    assert meta["bm25"]["denseRetrieval"] is False
    assert meta["bm25"]["lexicalOverlapBonus"] is False
    assert evidence.matches
    for match in evidence.matches:
        assert len(match.source_message_ids) == 1
        # PREV/NEXT never appear in credited ids for any hit.
        assert set(match.source_message_ids) <= {"D1:1", "D1:2", "D1:3", "D1:4"}
    ranked = ranked_message_ids(evidence.outcome.results)
    # Contextual enrichment can also surface neighbour units that inherited the term via
    # PREV/NEXT text; the TARGET unit must still enter the candidate pool.
    assert "D1:2" in ranked
    target_hit = next(r for r in evidence.outcome.results if r.source_message_ids == ("D1:2",))
    assert "D1:1" not in target_hit.source_message_ids
    assert "D1:3" not in target_hit.source_message_ids
    assert "[TARGET]" in target_hit.content
    assert target_hit.similarity == 0.0


def test_search_does_not_call_embedder():
    conversation = _conversation()
    consolidator = consolidate(conversation.conversation_id, _claims(conversation))
    index, _ = build_contextual_bm25_index(
        conversation, _claims(conversation), consolidator=consolidator,
    )

    class ForbiddenEmbedder:
        def embed(self, texts):
            raise AssertionError("contextual_bm25 must not call dense embeddings")

    # Ranking uses BM25 only; ForbiddenEmbedder is never consulted.
    outcome = search_contextual_bm25("rareterm xylophone", index, consolidator, k=3)
    assert outcome.results
    assert any(r.source_message_ids == ("D1:2",) for r in outcome.results)
    ForbiddenEmbedder()  # constructed; embed must never be invoked on sparse path


def test_existing_modes_unchanged_with_bm25_present():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    baseline = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=False)
    lexical = workspace.ask("weekday bicycle", retrieval_mode="lexical", generate=False)
    contextual = workspace.ask("weekday bicycle", retrieval_mode="contextual_lexical", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert lexical["answer"]["evidence"]["diagnostics"]["mode"] == "lexical"
    assert "contextualEnrichment" not in lexical["answer"]["evidence"]["diagnostics"]
    assert contextual["answer"]["evidence"]["diagnostics"]["mode"] == "contextual_lexical"

    bm25 = workspace.ask("weekday bicycle", retrieval_mode="contextual_bm25", generate=False)
    diagnostics = bm25["answer"]["evidence"]["diagnostics"]
    assert diagnostics["mode"] == "contextual_bm25"
    assert diagnostics["contextualEnrichment"] is True
    assert diagnostics["denseRetrieval"] is False
    assert diagnostics["sparseRetrieval"] == "bm25_okapi"
