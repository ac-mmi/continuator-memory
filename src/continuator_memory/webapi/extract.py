"""Import-time extraction wrappers. The LLM atomizer runs only when the caller asks."""

from __future__ import annotations

from continuator_memory.models import Conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import AtomizationResult, SemanticAtomizer, default_atomizer, memory_atomizer
from continuator_memory.v2.models import STATUS_CLAIMS, Claim, Provenance, SemanticMessage, claim_id
from continuator_memory.v2.pipeline import V2IngestReport, V2IngestionPipeline
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.validation import ValidatedWindow
from continuator_memory.v2.windowing import ContextWindow


# messages: one claim per message (no model); v2: semantic atomizer (every proposition);
# memories: V1 memory extraction (durable memories with a memory type).
EXTRACT_MODES = ("messages", "v2", "memories")


class PassThroughAtomizer:
    """One claim per target message. Used by tests and the no-LLM import option."""

    model_name = "passthrough"

    def atomize(self, window: ContextWindow, prior: object = ()) -> AtomizationResult:
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


def ingest_imported(
    conversation: Conversation,
    store: Store,
    *,
    extract: str = "messages",
    atomizer: SemanticAtomizer | PassThroughAtomizer | None = None,
) -> V2IngestReport:
    if extract not in EXTRACT_MODES:
        raise ValueError(f"extract must be one of {EXTRACT_MODES}")
    if atomizer is None:
        if extract == "v2":
            atomizer = default_atomizer()
        elif extract == "memories":
            atomizer = memory_atomizer()
        else:
            atomizer = PassThroughAtomizer()
    SemanticStore(store.connection)  # ensure V2 tables exist
    return V2IngestionPipeline(store, atomizer, embedder=None).ingest(conversation)
