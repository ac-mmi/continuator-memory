"""Search EvidenceUnits via the existing BM25 ``RetrievalUnit`` abstraction.

Conversation retrieval modes are untouched. This builds opaque ``RetrievalUnit``
records from persisted evidence and scores them with ``BM25Index``.

Document queries are gated with the product lexical ``STOP_WORDS`` set: stopword-only
(or otherwise empty-of-content-term) queries return no hits without running BM25.
"""

from __future__ import annotations

from continuator_memory.evidence.models import Document, EvidenceHit, EvidenceUnit
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.v2.bm25 import BM25Index, RetrievalUnit
from continuator_memory.v2.lexical import STOP_WORDS, tokenize


def document_query_content_tokens(query: str) -> list[str]:
    """Tokenize a document query and drop lexical stopwords (order preserved).

    Uses the same ``tokenize`` / ``STOP_WORDS`` conventions as product lexical overlap.
    Does not stem or apply length filters beyond tokenization.
    """
    tokens: list[str] = []
    for tok in tokenize(query or ""):
        if tok in STOP_WORDS:
            continue
        tokens.append(tok)
    return tokens


def units_to_retrieval_units(units: list[EvidenceUnit]) -> list[RetrievalUnit]:
    """Map evidence → BM25 units. ``id`` is ``evidence_id``; provenance stays in the store."""
    return [
        RetrievalUnit(
            id=unit.evidence_id,
            retrieval_content=unit.retrieval_text,
            # Caller-owned opaque ids: document + section for future gold matching.
            source_message_ids=(unit.document_id, unit.section_id, unit.evidence_id),
        )
        for unit in units
    ]


def search_evidence(
    store: EvidenceStore,
    query: str,
    *,
    top_k: int = 10,
    document_id: str | None = None,
) -> list[EvidenceHit]:
    """Lexical BM25 search over persisted EvidenceUnits (no model download).

    Queries with no remaining content terms after stopword filtering return ``[]``
    without building/scoring a BM25 index (successful empty result, not an error).
    When content terms remain, BM25 runs on those terms only so stopwords cannot
    contribute score.
    """
    content_tokens = document_query_content_tokens(query)
    if not content_tokens:
        return []

    if document_id is not None:
        units = store.get_units(document_id)
        documents = {document_id: store.get_document(document_id)}
    else:
        units = store.list_units()
        documents = {d.document_id: d for d in store.list_documents()}

    if not units:
        return []

    index = BM25Index(units_to_retrieval_units(units))
    # Re-join filtered tokens; sparse_tokenize yields the same content terms.
    hits = index.search(" ".join(content_tokens), top_k)
    by_id = {u.evidence_id: u for u in units}
    results: list[EvidenceHit] = []
    for hit in hits:
        unit = by_id.get(hit.unit.id)
        if unit is None:
            continue
        document = documents.get(unit.document_id)
        if document is None:
            document = store.get_document(unit.document_id)
        if document is None:
            # Should not happen if FK integrity holds; skip rather than invent provenance.
            continue
        results.append(
            EvidenceHit(unit=unit, document=document, score=hit.score, rank=hit.rank)
        )
    return results


def resolve_provenance(
    store: EvidenceStore, evidence_id: str
) -> tuple[EvidenceUnit, Document] | None:
    """Resolve ``evidence_id`` → unit + parent document (full provenance chain)."""
    unit = store.get_unit(evidence_id)
    if unit is None:
        return None
    document = store.get_document(unit.document_id)
    if document is None:
        return None
    return unit, document
