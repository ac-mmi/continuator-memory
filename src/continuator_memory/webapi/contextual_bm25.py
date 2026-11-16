"""``contextual_bm25`` — true sparse BM25 over contextual retrieval units.

Pipeline (standalone candidate generation; no dense score, overlap bonus, RRF, or rerank)::

    Query
      → general sparse tokenization
      → Okapi BM25 over ephemeral PREV/TARGET/NEXT retrieval text
      → ranked target evidence units (source_message_ids = target only)

Distinct from product ``lexical`` (dense + lexical overlap bonus) and from
``contextual_lexical`` (same enriched text, but dense+overlap scoring).
"""

from __future__ import annotations

from typing import Any, Iterable

from continuator_memory.models import Conversation
from continuator_memory.v2.bm25 import BM25Index, RetrievalUnit, build_index
from continuator_memory.v2.bprime_links import empty_trace
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.contextual_units import memories_with_contextual_retrieval_text
from continuator_memory.v2.memory import Memory, MemoryLink, MemorySource
from continuator_memory.v2.memory_retrieval import (
    QuestionIntent,
    RetrievalOutcome,
    RetrievalResult,
    VersionLink,
    question_intent,
)
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.source_retrieval import (
    SourceEvidence,
    resolve,
    select_sources,
    source_chunks,
)

MODE_CONTEXTUAL_BM25 = "contextual_bm25"


def _direct_links(mid: str, links: list[MemoryLink]) -> tuple[VersionLink, ...]:
    out: list[VersionLink] = []
    for link in links:
        if link.older_memory_id == mid:
            out.append(VersionLink(link.type, "newer", link.newer_memory_id, link.relation_id))
        elif link.newer_memory_id == mid:
            out.append(VersionLink(link.type, "older", link.older_memory_id, link.relation_id))
    return tuple(dict.fromkeys(out))


def contextual_retrieval_units(
    conversation: Conversation,
    memories: Iterable[Memory],
    sources: Iterable[MemorySource],
) -> list[RetrievalUnit]:
    """Build BM25 units: enriched retrieval text, target-only ``source_message_ids``."""
    by_memory: dict[str, list[MemorySource]] = {}
    for source in sources:
        by_memory.setdefault(source.memory_id, []).append(source)
    enriched = memories_with_contextual_retrieval_text(conversation, memories)
    units: list[RetrievalUnit] = []
    for memory in enriched:
        # Provenance from original consolidator sources (target message), never PREV/NEXT ids.
        own = sorted(by_memory.get(memory.memory_id, ()), key=lambda s: (s.position, s.claim_id))
        source_ids = tuple(dict.fromkeys(s.message_id for s in own))
        units.append(
            RetrievalUnit(
                id=memory.memory_id,
                retrieval_content=memory.content,
                source_message_ids=source_ids,
            )
        )
    return units


def build_contextual_bm25_index(
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation] = (),
    *,
    consolidator=None,
) -> tuple[BM25Index, Any]:
    """Return ``(BM25Index, consolidator)`` for contextual units of one conversation."""
    if consolidator is None:
        consolidator = consolidate(conversation.conversation_id, claims, relations)
    units = contextual_retrieval_units(
        conversation,
        consolidator.memories.values(),
        consolidator.sources,
    )
    return build_index(units), consolidator


def _result_from_memory(
    rank: int,
    memory: Memory,
    *,
    retrieval_content: str,
    score: float,
    sources: tuple[MemorySource, ...],
    links: tuple[VersionLink, ...],
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        memory_id=memory.memory_id,
        conversation_id=memory.conversation_id,
        content=retrieval_content,
        similarity=0.0,
        adjustment=0.0,
        score=score,
        status=memory.status,
        time_class=getattr(memory, "time_class", "") or "",
        event_time=getattr(memory, "event_time", None),
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        superseded_by=memory.superseded_by,
        sources=sources,
        links=links,
        context_similarity=None,
    )


def search_contextual_bm25(
    question: str,
    index: BM25Index,
    consolidator: Any,
    *,
    k: int,
) -> RetrievalOutcome:
    """Rank consolidator memories by BM25 over contextual unit text."""
    if k < 1:
        raise ValueError("k must be >= 1")
    intent: QuestionIntent = question_intent(question)
    link_list = list(consolidator.links)
    hits = index.search(question, k)
    results: list[RetrievalResult] = []
    for hit in hits:
        memory = consolidator.memories.get(hit.unit.id)
        if memory is None:
            continue
        sources = tuple(
            sorted(consolidator.sources_of(memory.memory_id), key=lambda s: (s.position, s.claim_id))
        )
        results.append(
            _result_from_memory(
                hit.rank,
                memory,
                retrieval_content=hit.unit.retrieval_content,
                score=hit.score,
                sources=sources,
                links=_direct_links(memory.memory_id, link_list),
            )
        )
    return RetrievalOutcome(question, intent, tuple(results))


def retrieve_contextual_bm25_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    k: int,
    budget_chars: int,
    *,
    consolidator=None,
    index: BM25Index | None = None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Standalone BM25 candidate generation; no embedder / dense path involved."""
    if index is None:
        index, consolidator = build_contextual_bm25_index(
            conversation, claims, relations, consolidator=consolidator,
        )
    elif consolidator is None:
        consolidator = consolidate(conversation.conversation_id, claims, relations)
    outcome = search_contextual_bm25(question, index, consolidator, k=k)
    matches = tuple(resolve(result, conversation) for result in outcome.results)
    selection = select_sources(matches, budget_chars)
    evidence = SourceEvidence(
        question,
        outcome,
        matches,
        selection,
        source_chunks(selection, matches, conversation),
        bprime=empty_trace(enabled=False),
        n_seed_matches=len(matches),
    )
    meta = {
        "bm25": {
            "algorithm": "okapi",
            "k1": index.k1,
            "b": index.b,
            "epsilon": index.epsilon,
            "nDocuments": index.n_docs,
            "avgdl": index.avgdl,
            "tokenizer": "continuator_memory.v2.lexical.tokenize",
            "representation": "contextual_prev_target_next",
            "denseRetrieval": False,
            "lexicalOverlapBonus": False,
        }
    }
    return evidence, meta
