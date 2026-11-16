"""Focused tests for Option B contextual retrieval units (no LoCoMo full eval)."""

from __future__ import annotations

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.contextual_units import (
    build_contextual_retrieval_text,
    memories_with_contextual_retrieval_text,
    neighbour_messages,
)
from continuator_memory.v2.memory import Memory, STATUS_CURRENT
from continuator_memory.v2.memory_retrieval import RetrievalConfig
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.v2.source_retrieval import retrieve_source_evidence
from continuator_memory.webapi.locomo_report import ranked_message_ids, retrieval_config
from continuator_memory.webapi.multiquery import build_memory_retriever
from continuator_memory.webapi.workspace import RETRIEVAL_MODES, Workspace, normalize_retrieval_mode
from tests.helpers import HashingEmbedder


def _msg(mid: str, position: int, text: str, timestamp: str | None) -> Message:
    return Message(mid, "conv", position, "user", text, timestamp)


def _three_turn_conversation(*, session_split_after: int | None = None) -> Conversation:
    """Three turns; optional session boundary after index ``session_split_after`` (0-based)."""
    stamps = ["2023-01-01T10:00:00", "2023-01-01T10:00:00", "2023-01-01T10:00:00"]
    if session_split_after is not None:
        for i in range(session_split_after + 1, 3):
            stamps[i] = "2023-01-02T10:00:00"
    messages = (
        _msg("D1:1", 0, "Alice: hello previous", stamps[0]),
        _msg("D1:2", 1, "Bob: target facts about swimming", stamps[1]),
        _msg("D1:3", 2, "Alice: next acknowledgement", stamps[2]),
    )
    return Conversation("conv", messages, title="test")


def _memory_for(message: Message) -> Memory:
    return Memory(
        memory_id=f"mem:{message.message_id}:c0",
        conversation_id=message.conversation_id,
        content=message.content,
        status=STATUS_CURRENT,
        claim_id=f"{message.message_id}:c0",
        message_id=message.message_id,
        position=message.position,
        created_at=message.timestamp,
    )


def test_middle_message_gets_prev_target_next():
    conversation = _three_turn_conversation()
    memory = _memory_for(conversation.messages[1])
    text = build_contextual_retrieval_text(conversation, memory)
    assert text.startswith("[PREV] Alice: hello previous")
    assert "[TARGET] Bob: target facts about swimming" in text
    assert text.endswith("[NEXT] Alice: next acknowledgement")
    prev, target, nxt = neighbour_messages(conversation, "D1:2")
    assert [m.message_id for m in prev] == ["D1:1"]
    assert target is not None and target.message_id == "D1:2"
    assert [m.message_id for m in nxt] == ["D1:3"]


def test_first_message_omits_prev_last_omits_next():
    conversation = _three_turn_conversation()
    first = build_contextual_retrieval_text(conversation, _memory_for(conversation.messages[0]))
    last = build_contextual_retrieval_text(conversation, _memory_for(conversation.messages[2]))
    assert "[PREV]" not in first
    assert first.startswith("[TARGET] Alice: hello previous")
    assert "[NEXT] Bob: target facts about swimming" in first
    assert last.startswith("[PREV] Bob: target facts about swimming")
    assert "[TARGET] Alice: next acknowledgement" in last
    assert "[NEXT]" not in last


def test_cross_session_neighbours_excluded():
    # Session boundary between D1:1 and D1:2: middle has no prev; first has no next.
    conversation = _three_turn_conversation(session_split_after=0)
    middle = build_contextual_retrieval_text(conversation, _memory_for(conversation.messages[1]))
    assert "[PREV]" not in middle
    assert "[TARGET] Bob: target facts about swimming" in middle
    assert "[NEXT] Alice: next acknowledgement" in middle
    first = build_contextual_retrieval_text(conversation, _memory_for(conversation.messages[0]))
    assert "[NEXT]" not in first
    assert first == "[TARGET] Alice: hello previous"


def test_enrichment_does_not_mutate_original_memory():
    conversation = _three_turn_conversation()
    original = _memory_for(conversation.messages[1])
    before = original.content
    copies = memories_with_contextual_retrieval_text(conversation, [original])
    assert original.content == before
    assert copies[0].content != before
    assert "[TARGET]" in copies[0].content
    assert copies[0].message_id == "D1:2"
    assert copies[0].memory_id == original.memory_id


def test_contextual_index_preserves_target_only_source_ids():
    conversation = _three_turn_conversation()
    claims = [
        Claim(f"{m.message_id}:c0", m.content, Provenance.of(m))
        for m in conversation.messages
    ]
    consolidator = consolidate(conversation.conversation_id, claims)
    # Sanity: stored sources remain 1:1 with targets.
    assert {s.message_id for s in consolidator.sources} == {"D1:1", "D1:2", "D1:3"}
    for memory in consolidator.memories.values():
        assert memory.content.startswith(("Alice:", "Bob:"))
        assert "[TARGET]" not in memory.content

    retriever = build_memory_retriever(
        conversation,
        claims,
        (),
        HashingEmbedder(),
        RetrievalConfig(lexical_weight=0.25, context_weight=0.0),
        consolidator=consolidator,
        contextual_enrichment=True,
    )
    outcome = retriever.search("swimming", k=3, conversation_id="conv")
    assert outcome.results
    for result in outcome.results:
        assert len(result.source_message_ids) == 1
        assert result.source_message_ids[0] == consolidator.memories[result.memory_id].message_id
        # Indexed content used for retrieval is enriched…
        assert "[TARGET]" in result.content
        # …but consolidator / identity mapping is still the target dia_id only.
        assert consolidator.memories[result.memory_id].message_id in {"D1:1", "D1:2", "D1:3"}
        assert "[TARGET]" not in consolidator.memories[result.memory_id].content

    ranked = ranked_message_ids(outcome.results)
    assert set(ranked) <= {"D1:1", "D1:2", "D1:3"}
    # Neighbours must not be credited from a single enriched hit.
    target_hit = next(r for r in outcome.results if r.source_message_ids == ("D1:2",))
    assert "D1:1" not in target_hit.source_message_ids
    assert "D1:3" not in target_hit.source_message_ids


def test_retrieve_source_evidence_contextual_keeps_target_sources():
    conversation = _three_turn_conversation()
    claims = [
        Claim(f"{m.message_id}:c0", m.content, Provenance.of(m))
        for m in conversation.messages
    ]
    evidence = retrieve_source_evidence(
        "swimming",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        k=3,
        budget_chars=10_000,
        config=RetrievalConfig(lexical_weight=0.25, context_weight=0.5),  # weight must be forced off
        contextual_enrichment=True,
    )
    for match in evidence.matches:
        assert len(match.source_message_ids) == 1
    # Dual-vector path disabled even if caller passed context_weight>0.
    assert evidence.outcome.results[0].context_similarity is None


def test_mode_registration_and_retrieval_config():
    assert "contextual_lexical" in RETRIEVAL_MODES
    assert normalize_retrieval_mode("contextual_lexical") == "contextual_lexical"
    cfg = retrieval_config("contextual_lexical")
    lex = retrieval_config("lexical")
    assert cfg is not None and lex is not None
    assert cfg.lexical_weight == lex.lexical_weight
    assert cfg.context_weight == 0.0
    # Existing modes still resolve.
    assert normalize_retrieval_mode("lexical") == "lexical"
    assert normalize_retrieval_mode("hybrid") == "lexical"
    assert normalize_retrieval_mode("baseline") == "baseline"


def test_existing_modes_unaffected_by_contextual_builder():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    baseline = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=False)
    lexical = workspace.ask("weekday bicycle", retrieval_mode="lexical", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert lexical["answer"]["evidence"]["diagnostics"]["mode"] == "lexical"
    assert "contextualEnrichment" not in lexical["answer"]["evidence"]["diagnostics"]

    contextual = workspace.ask("weekday bicycle", retrieval_mode="contextual_lexical", generate=False)
    diagnostics = contextual["answer"]["evidence"]["diagnostics"]
    assert diagnostics["mode"] == "contextual_lexical"
    assert diagnostics["contextualEnrichment"] is True
    assert diagnostics["contextWeight"] == 0.0
