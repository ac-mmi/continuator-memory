"""Model-agnostic MemoryExtractor contract and factory."""

from __future__ import annotations

import json

from continuator_memory.models import parse_conversation
from continuator_memory.v2.atomizer import (
    ATOMIZATION_PROTOCOL,
    MEMORY_PROTOCOL,
    AtomizationRequestError,
    SemanticAtomizer,
    V2Settings,
    default_atomizer,
    memory_atomizer,
)
from continuator_memory.v2.extractors import (
    EXTRACTOR_OLLAMA,
    EXTRACTOR_PASSTHROUGH,
    MemoryExtractor,
    PassThroughExtractor,
    build_extractor,
    extractor_for_import_mode,
    ollama_memory_extractor,
)
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.windowing import build_windows
from tests.unit.test_v2_pipeline import CONVERSATION, SCRIPT, ScriptedAtomizerClient, c


def test_passthrough_and_semantic_atomizer_satisfy_memory_extractor_protocol():
    assert isinstance(PassThroughExtractor(), MemoryExtractor)
    client = ScriptedAtomizerClient({"m1": {"messages": []}})
    assert isinstance(SemanticAtomizer(client), MemoryExtractor)


def test_build_extractor_ollama_uses_injected_client_not_network():
    client = ScriptedAtomizerClient({
        "m1": {"messages": [
            {"source_message_id": "m1", "status": "claims",
             "claims": [c("m1:c1", "Leah lives in Madrid.", ["Leah", "Madrid"])],
             "explicit_relations": []},
            {"source_message_id": "m2", "status": "no_durable_assertion", "claims": [], "explicit_relations": []},
        ]},
    })
    extractor = build_extractor(EXTRACTOR_OLLAMA, client=client)
    assert isinstance(extractor, SemanticAtomizer)
    window = build_windows(parse_conversation(CONVERSATION), 2, 2)[0]
    result = extractor.atomize(window)
    assert result.validated.error is None
    assert [cl.claim_id for cl in result.validated.claims] == ["m1:c1"]
    assert client.calls  # went through JSONCompletionClient, not Ollama HTTP


def test_build_extractor_passthrough_needs_no_client():
    extractor = build_extractor(EXTRACTOR_PASSTHROUGH)
    assert isinstance(extractor, PassThroughExtractor)
    window = build_windows(parse_conversation(CONVERSATION), 2, 2)[0]
    result = extractor.atomize(window)
    assert [cl.content for cl in result.validated.claims] == [
        "My sister Leah lives in Madrid.",
        "Nice! Will you visit her?",
    ]


def test_pipeline_accepts_injected_memory_extractor(store):
    client = ScriptedAtomizerClient(SCRIPT)
    extractor: MemoryExtractor = SemanticAtomizer(client)
    report = V2IngestionPipeline(store, extractor, targets_per_window=2, context_before=2).ingest(
        parse_conversation(CONVERSATION)
    )
    assert report.claims == 5 and report.failed_windows == 0
    assert MemoryStore(store.connection).has_snapshot("trip")
    assert len(store.get_conversation("trip").messages) == 4


def test_extractor_for_import_mode_messages_is_passthrough():
    assert isinstance(extractor_for_import_mode("messages"), PassThroughExtractor)


def test_extractor_for_import_mode_respects_settings_extractor_backend():
    settings = V2Settings(extractor=EXTRACTOR_PASSTHROUGH)
    assert isinstance(extractor_for_import_mode("v2", settings=settings), PassThroughExtractor)
    assert isinstance(extractor_for_import_mode("memories", settings=settings), PassThroughExtractor)


def test_extractor_for_import_mode_explicit_atomizer_wins():
    custom = PassThroughExtractor()
    assert extractor_for_import_mode("v2", atomizer=custom) is custom


def test_default_and_memory_atomizer_factories_remain_ollama_semantic():
    settings = V2Settings(atomizer_model="test-model", ollama_host="http://example.invalid")
    assert isinstance(default_atomizer(settings), SemanticAtomizer)
    assert default_atomizer(settings).model_name == "test-model"
    assert isinstance(memory_atomizer(settings), SemanticAtomizer)
    assert memory_atomizer(settings).protocol is MEMORY_PROTOCOL
    assert ollama_memory_extractor(settings).protocol is ATOMIZATION_PROTOCOL


def test_build_extractor_rejects_unknown_backend():
    try:
        build_extractor("memreader")
    except ValueError as exc:
        assert "unknown extractor" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_semantic_atomizer_request_error_becomes_failed_window():
    class Boom:
        model_name = "boom"

        def complete_json(self, *args, **kwargs):
            raise AtomizationRequestError("offline")

    window = build_windows(parse_conversation(CONVERSATION), 2, 2)[0]
    result = SemanticAtomizer(Boom()).atomize(window)
    assert result.validated.error == "offline"
    assert {m.status for m in result.validated.messages} == {"failed"}


def test_ollama_extractor_output_is_atomization_result_with_validated_claims():
    """Canonical contract: AtomizationResult.validated feeds claim storage unchanged."""
    from continuator_memory.v2.atomizer import AtomizationResult

    raw = json.dumps({"messages": [
        {"source_message_id": "m1", "status": "claims",
         "claims": [c("m1:c1", "Leah lives in Madrid.", ["Leah", "Madrid"])],
         "explicit_relations": []},
        {"source_message_id": "m2", "status": "no_durable_assertion", "claims": [], "explicit_relations": []},
    ]})
    result = build_extractor(EXTRACTOR_OLLAMA, client=ScriptedAtomizerClient({"m1": raw})).atomize(
        build_windows(parse_conversation(CONVERSATION), 2, 2)[0]
    )
    assert isinstance(result, AtomizationResult)
    assert result.validated.claims[0].claim_id == "m1:c1"
    assert result.validated.claims[0].content == "Leah lives in Madrid."
