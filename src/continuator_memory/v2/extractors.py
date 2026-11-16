"""Model-agnostic extraction interface for the V2 product pipeline.

The ingestion pipeline depends only on ``MemoryExtractor``: a component that turns a
``ContextWindow`` (plus optional prior claims) into an ``AtomizationResult``. Validation,
claim storage and consolidation never see the backend that produced the result.

Built-in implementations:

* ``ollama`` — ``SemanticAtomizer`` over ``OllamaChatClient`` (default production path)
* ``passthrough`` — one claim per target message; no model call

A future HuggingFace / MemReader / HTTP adapter implements the same protocol and is
selected by ``build_extractor`` (or injected directly into ``V2IngestionPipeline``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from continuator_memory.v2.atomizer import (
    ATOMIZATION_PROTOCOL,
    MEMORY_PROTOCOL,
    AtomizationResult,
    ExtractionProtocol,
    JSONCompletionClient,
    OllamaChatClient,
    SemanticAtomizer,
    V2Settings,
)
from continuator_memory.v2.models import STATUS_CLAIMS, Claim, Provenance, SemanticMessage, claim_id
from continuator_memory.v2.validation import ValidatedWindow
from continuator_memory.v2.windowing import ContextWindow

EXTRACTOR_OLLAMA = "ollama"
EXTRACTOR_PASSTHROUGH = "passthrough"
EXTRACTOR_TYPES = (EXTRACTOR_OLLAMA, EXTRACTOR_PASSTHROUGH)

# Import / product modes: which *instruction protocol* to use (orthogonal to backend).
EXTRACT_MODE_MESSAGES = "messages"
EXTRACT_MODE_V2 = "v2"
EXTRACT_MODE_MEMORIES = "memories"
EXTRACT_MODES = (EXTRACT_MODE_MESSAGES, EXTRACT_MODE_V2, EXTRACT_MODE_MEMORIES)


@runtime_checkable
class MemoryExtractor(Protocol):
    """Product extraction contract: window + prior claims → validated atomization result."""

    model_name: str

    def atomize(self, window: ContextWindow, prior: Sequence[Claim] = ()) -> AtomizationResult: ...


class PassThroughExtractor:
    """One claim per non-empty target message. No model call; used for tests and default import."""

    model_name = "passthrough"

    def atomize(self, window: ContextWindow, prior: Sequence[Claim] = ()) -> AtomizationResult:
        messages = []
        for message in window.targets:
            content = message.content.strip()
            if not content:
                messages.append(
                    SemanticMessage(Provenance.of(message), message.content, STATUS_CLAIMS, (), ())
                )
                continue
            claim = Claim(claim_id(message.message_id, 0), content, Provenance.of(message))
            messages.append(
                SemanticMessage(Provenance.of(message), message.content, STATUS_CLAIMS, (claim,), ())
            )
        return AtomizationResult(ValidatedWindow(window, messages, raw="passthrough"), "", 0.0)


# Backwards-compatible name used by older tests and docs.
PassThroughAtomizer = PassThroughExtractor


def build_extractor(
    extractor: str = EXTRACTOR_OLLAMA,
    *,
    protocol: ExtractionProtocol | None = None,
    settings: V2Settings | None = None,
    client: JSONCompletionClient | None = None,
) -> MemoryExtractor:
    """Construct a ``MemoryExtractor`` by backend name.

    ``extractor`` selects the implementation (``ollama``, ``passthrough``). Model identity for
    the Ollama backend comes from ``settings.atomizer_model`` / ``CONTINUATOR_V2_ATOMIZER_MODEL``,
    not from this name. Pass ``client`` to inject a mock or non-Ollama ``JSONCompletionClient``.
    """
    kind = extractor.strip().lower()
    if kind == EXTRACTOR_PASSTHROUGH:
        return PassThroughExtractor()
    if kind == EXTRACTOR_OLLAMA:
        s = settings or V2Settings.from_env()
        json_client = client or OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, s.num_predict)
        return SemanticAtomizer(json_client, protocol or ATOMIZATION_PROTOCOL)
    raise ValueError(f"unknown extractor {extractor!r}; expected one of {EXTRACTOR_TYPES}")


def extractor_for_import_mode(
    extract: str,
    *,
    settings: V2Settings | None = None,
    client: JSONCompletionClient | None = None,
    atomizer: MemoryExtractor | None = None,
) -> MemoryExtractor:
    """Resolve the extractor for a product import mode (``messages`` / ``v2`` / ``memories``).

    ``messages`` always uses passthrough. ``v2`` / ``memories`` use ``settings.extractor``
    (default ``ollama``) with the matching instruction protocol. An explicit ``atomizer`` wins.
    """
    if atomizer is not None:
        return atomizer
    if extract not in EXTRACT_MODES:
        raise ValueError(f"extract must be one of {EXTRACT_MODES}")
    if extract == EXTRACT_MODE_MESSAGES:
        return PassThroughExtractor()
    s = settings or V2Settings.from_env()
    protocol = MEMORY_PROTOCOL if extract == EXTRACT_MODE_MEMORIES else ATOMIZATION_PROTOCOL
    return build_extractor(s.extractor, protocol=protocol, settings=s, client=client)


def ollama_memory_extractor(
    settings: V2Settings | None = None,
    *,
    protocol: ExtractionProtocol = ATOMIZATION_PROTOCOL,
    client: JSONCompletionClient | None = None,
) -> MemoryExtractor:
    """Default production extractor: SemanticAtomizer + Ollama (typically Qwen)."""
    return build_extractor(EXTRACTOR_OLLAMA, protocol=protocol, settings=settings, client=client)
