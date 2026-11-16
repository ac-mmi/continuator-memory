import json

import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.context import (
    SPEAKER_AMBIGUOUS,
    SPEAKER_NAMED,
    SPEAKER_UNNAMED,
    ConversationMessages,
    SourceContextError,
    SourceWindow,
    SpeakerRegistry,
    contextual_memories,
    contextualize_claims,
    format_contextual_memory,
    global_memory_id,
    resolve_window,
    windows_by_message,
)
from continuator_memory.v2.context_store import (
    ContextStore,
    contextualize_stored,
    raw_messages,
    stored_contextual_memories,
    stored_windows,
)
from continuator_memory.v2.memory_retrieval import MemoryRetriever
from continuator_memory.v2.memory_store import MemoryStore, consolidate_stored
from continuator_memory.v2.models import SUPPORT_IDENTITY, Claim, Provenance
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.windowing import build_windows


class ClaimsClient:
    """Scripted atomizer: returns the listed claim contents for every target message it is given."""

    model_name = "scripted-atomizer"

    def __init__(self, claims):
        self.claims = claims

    def complete_json(self, system, prompt, schema, examples=()):
        ids = schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"]
        entries = []
        for mid in ids:
            claims = [{"claim_id": f"{mid}:c{i}", "content": content, "entities": [], "qualifiers": [],
                       "temporal": {"expression": "", "resolved_date": ""}, "polarity": "positive",
                       "claim_type": "", "source_message_id": mid}
                      for i, content in enumerate(self.claims.get(mid, ()), 1)]
            entries.append({"source_message_id": mid, "status": "claims" if claims else "no_durable_assertion",
                            "claims": claims, "explicit_relations": []})
        return json.dumps({"messages": entries})


def conversation(cid, *messages):
    return parse_conversation({"conversation_id": cid, "messages": [
        {"id": f"m{i}", "role": role, "content": content} for i, (role, content) in enumerate(messages, 1)]})


def ingest(store, conv, claims):
    pipeline = V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient(claims)), targets_per_window=2,
                                   context_before=2)
    report = pipeline.ingest(conv)
    assert not report.issues, report.issues
    consolidate_stored(store.connection, conv.conversation_id)
    contextualize_stored(store.connection, conv.conversation_id)
    return {cm.memory.claim_id: cm for cm in stored_contextual_memories(store.connection, conv.conversation_id)}


SAM = conversation("sam-home", ("user", "Hello, I'm Sam."), ("assistant", "Nice to meet you, Sam!"),
                   ("user", "My dog is Pepper."))
ALEX = conversation("alex-home", ("user", "Hi, I'm Alex."), ("assistant", "Nice to meet you, Alex!"),
                    ("user", "My dog is Biscuit."))
DOGS = {"m3": ["The user's dog is Pepper."]}


# ---- memory identity --------------------------------------------------------


def test_memory_id_collides_across_conversations_and_global_id_does_not(store):
    sam = ingest(store, SAM, DOGS)["m3:c1"]
    alex = ingest(store, ALEX, {"m3": ["The user's dog is Biscuit."]})["m3:c1"]
    assert sam.memory.memory_id == alex.memory.memory_id == "mem:m3:c1"  # the Phase 4 collision
    assert (sam.global_id, alex.global_id) == ("sam-home/mem:m3:c1", "alex-home/mem:m3:c1")
    assert sam.global_id != alex.global_id


def test_global_memory_id_is_injective_for_separator_characters():
    pairs = [("a:b", "c"), ("a", "b:c"), ("a/b", "c"), ("a", "b/c"), ("a%2Fb", "c")]
    assert len({global_memory_id(*p) for p in pairs}) == len(pairs)


def test_retrieval_index_keeps_both_colliding_memories(embedder, store):
    ingest(store, SAM, DOGS)
    ingest(store, ALEX, {"m3": ["The user's dog is Biscuit."]})
    retriever = MemoryRetriever(embedder)
    for cid in ("sam-home", "alex-home"):
        retriever.index_store(MemoryStore(store.connection), cid)
    results = retriever.retrieve("dog", k=5)
    assert sorted((r.conversation_id, r.memory_id) for r in results) == [
        ("alex-home", "mem:m3:c1"), ("sam-home", "mem:m3:c1")]


# ---- speaker identity -------------------------------------------------------


def test_speakers_in_two_conversations_stay_distinguishable(store):
    sam = ingest(store, SAM, DOGS)["m3:c1"]
    alex = ingest(store, ALEX, {"m3": ["The user's dog is Biscuit."]})["m3:c1"]
    assert sam.memory.content == "The user's dog is Pepper."  # claim/memory content unchanged
    assert sam.contextual_content == "Sam's dog is Pepper."
    assert alex.contextual_content == "Alex's dog is Biscuit."
    assert (sam.speaker.role, sam.speaker.name, sam.speaker.status, sam.speaker.evidence) == (
        "user", "Sam", SPEAKER_NAMED, ("m1",))
    assert alex.speaker.name == "Alex" and "Sam" not in alex.contextual_content
    assert [s.name for s in sam.context.resolved_speakers] == ["Sam"]


def test_user_and_assistant_names_in_one_conversation_are_not_mixed(store):
    conv = conversation("pets", ("user", "Hello, I'm Sam."), ("assistant", "Hello Sam, I'm Nova."),
                        ("user", "My dog is Pepper."), ("assistant", "My dog is Biscuit."))
    out = ingest(store, conv, {"m3": ["The user's dog is Pepper."], "m4": ["The assistant's dog is Biscuit."]})
    assert out["m3:c1"].contextual_content == "Sam's dog is Pepper."
    assert out["m4:c1"].contextual_content == "Nova's dog is Biscuit."
    assert (out["m4:c1"].speaker.role, out["m4:c1"].speaker.name) == ("assistant", "Nova")


def test_unnamed_user_keeps_frame_wording(store):
    out = ingest(store, conversation("anon", ("user", "My cat is Miso.")), {"m1": ["The user's cat is Miso."]})
    cm = out["m1:c1"]
    assert cm.contextual_content == "The user's cat is Miso."
    assert (cm.speaker.name, cm.speaker.status) == (None, SPEAKER_UNNAMED)


def test_capitalized_word_after_im_is_not_a_name_without_corroboration(store):
    conv = conversation("demonym", ("user", "I'm Spanish. My bike is red."))
    cm = ingest(store, conv, {"m1": ["The user's bike is red."]})["m1:c1"]
    assert cm.contextual_content == "The user's bike is red." and cm.speaker.status == SPEAKER_UNNAMED


def test_two_names_for_one_role_stay_ambiguous(store):
    conv = conversation("shared", ("user", "Hi, I'm Sam."), ("user", "My name is Jordan, I share this account."),
                        ("user", "My dog is Pepper."))
    cm = ingest(store, conv, {"m3": ["The user's dog is Pepper."]})["m3:c1"]
    assert cm.contextual_content == "The user's dog is Pepper."
    assert cm.speaker.status == SPEAKER_AMBIGUOUS and cm.speaker.evidence == ("m1", "m2")


def test_name_stated_after_the_claim_is_not_used(store):
    conv = conversation("late", ("user", "My dog is Pepper."), ("user", "By the way, my name is Sam."))
    cm = ingest(store, conv, {"m1": ["The user's dog is Pepper."]})["m1:c1"]
    assert cm.contextual_content == "The user's dog is Pepper." and cm.speaker.status == SPEAKER_UNNAMED


def test_literal_user_in_the_source_is_not_rewritten(store):
    conv = conversation("app", ("user", "Hi, I'm Sam."), ("user", "In my app the user must log in first."))
    cm = ingest(store, conv, {"m2": ["In the user's app the user must log in first."]})
    assert cm["m2:c1"].contextual_content == "In the user's app the user must log in first."


def test_speaker_registry_strong_forms_and_other_roles():
    conv = conversation("r", ("assistant", "Call me Nova."), ("user", "My name is Sam Carter."))
    registry = SpeakerRegistry(conv.messages)
    assert registry.speaker("user", 0).status == SPEAKER_UNNAMED  # only the assistant has spoken
    assert (registry.speaker("user", 1).name, registry.speaker("assistant", 1).name) == ("Sam Carter", "Nova")


# ---- references -------------------------------------------------------------


LAPTOP = conversation("laptop", ("user", "Hi, I'm Sam."), ("user", "I bought a new laptop."),
                      ("user", "It is a ThinkPad."))


def test_unresolved_pronoun_is_flagged_and_window_holds_its_referent(store):
    out = ingest(store, LAPTOP, {"m2": ["The user bought a new laptop."], "m3": ["It is a ThinkPad."]})
    cm = out["m3:c1"]
    assert cm.contextual_content == "It is a ThinkPad." and cm.context.unresolved_reference
    ctx = cm.evidence(store)[0]
    assert [(m.message_id, m.role, m.content) for m in ctx.messages] == [
        ("m1", "user", "Hi, I'm Sam."), ("m2", "user", "I bought a new laptop."), ("m3", "user", "It is a ThinkPad.")]
    assert cm.speaker.name == "Sam"
    assert out["m2:c1"].contextual_content == "Sam bought a new laptop."


def test_atomizer_resolved_reference_becomes_a_self_contained_memory(store):
    out = ingest(store, LAPTOP, {"m2": ["The user bought a new laptop."],
                                 "m3": ["The user's new laptop is a ThinkPad."]})
    cm = out["m3:c1"]
    claim = next(c for c in SemanticStore(store.connection).get_claims("laptop") if c.claim_id == "m3:c1")
    assert claim.support == SUPPORT_IDENTITY and claim.resolved_from == ("m2",)
    assert cm.contextual_content == "Sam's new laptop is a ThinkPad."
    assert not cm.context.unresolved_reference and cm.context.resolved_from == ("m2",)
    assert "m2" in cm.context.window.message_ids


# ---- assistant / user attribution -------------------------------------------


ADVICE = conversation("advice", ("user", "Hi, I'm Priya. I live in Lisbon."),
                      ("assistant", "Lisbon has excellent public transport."),
                      ("assistant", "I recommend the Gulbenkian Museum."))
ADVICE_CLAIMS = {"m1": ["The user lives in Lisbon."], "m2": ["Lisbon has excellent public transport."],
                 "m3": ["The assistant recommends the Gulbenkian Museum."]}


def test_assistant_statements_are_attributed_to_the_assistant(store):
    out = ingest(store, ADVICE, ADVICE_CLAIMS)
    assert out["m1:c1"].contextual_content == "Priya lives in Lisbon."
    transport, museum = out["m2:c1"], out["m3:c1"]
    assert transport.contextual_content == "The assistant stated: Lisbon has excellent public transport."
    assert museum.contextual_content == "The assistant recommends the Gulbenkian Museum."
    for cm in (transport, museum):
        assert cm.speaker.role == "assistant" and "Priya" not in cm.contextual_content
        assert "user" not in cm.contextual_content.lower()


# ---- source preservation and round trip -------------------------------------


def test_every_memory_recovers_its_ordered_source_window(store):
    memories = [*ingest(store, ADVICE, ADVICE_CLAIMS).values(), *ingest(store, SAM, DOGS).values()]
    assert memories
    for cm in memories:
        for source, ctx in zip(cm.sources, cm.evidence(store)):
            raw = store.get_conversation(cm.memory.conversation_id).messages
            assert ctx.messages == raw[ctx.window.start:ctx.window.end + 1]
            assert ctx.messages[-1].message_id == source.message_id == cm.memory.message_id
            assert [m.position for m in ctx.messages] == sorted(m.position for m in ctx.messages)


def test_stored_windows_are_the_atomization_windows(store):
    ingest(store, ADVICE, ADVICE_CLAIMS)
    assert stored_windows(store.connection, "advice") == windows_by_message(build_windows(ADVICE, 2, 2))
    assert stored_windows(store.connection, "advice")["m3"] == SourceWindow("advice", 0, 2, ("m1", "m2", "m3"))


def test_round_trip_is_lossless_and_deterministic(store):
    ingest(store, LAPTOP, {"m2": ["The user bought a new laptop."], "m3": ["It is a ThinkPad."]})
    stored = ContextStore(store.connection).get_contexts("laptop")
    recomputed = contextualize_claims(raw_messages(store.connection, "laptop"),
                                      SemanticStore(store.connection).get_claims("laptop"),
                                      stored_windows(store.connection, "laptop"))
    assert stored == recomputed and list(stored) == list(recomputed)

    def snapshot():
        s = Store(":memory:")
        try:
            out = ingest(s, LAPTOP, {"m2": ["The user bought a new laptop."], "m3": ["It is a ThinkPad."]})
            return json.dumps([[cm.to_dict(), [c.to_dict() for c in cm.evidence(s)]] for cm in out.values()])
        finally:
            s.close()

    assert snapshot() == snapshot()


def test_in_memory_path_matches_stored_path(store):
    stored = ingest(store, ADVICE, ADVICE_CLAIMS)
    claims = SemanticStore(store.connection).get_claims("advice")
    contexts = contextualize_claims(ADVICE.messages, claims, windows_by_message(build_windows(ADVICE, 2, 2)))
    c = consolidate("advice", claims)
    in_memory = contextual_memories(c.memories.values(), c.sources, contexts)
    assert [cm.to_dict() for cm in in_memory] == [cm.to_dict() for cm in stored.values()]
    lookup = ConversationMessages([ADVICE])
    assert [x.evidence(lookup) for x in in_memory] == [x.evidence(store) for x in stored.values()]


def test_tampered_reference_is_rejected(store):
    ingest(store, SAM, DOGS)
    with pytest.raises(SourceContextError):
        resolve_window(SourceWindow("sam-home", 0, 2, ("m1", "m3", "m2")), store)
    with pytest.raises(SourceContextError):
        resolve_window(SourceWindow("alex-home", 0, 2, ("m1", "m2", "m3")), store)


def test_missing_window_is_an_error():
    claims_conv = conversation("x", ("user", "My cat is Miso."))
    claim = Claim("m1:c1", "The user's cat is Miso.", Provenance.of(claims_conv.messages[0]))
    with pytest.raises(SourceContextError):
        contextualize_claims(claims_conv.messages, [claim], {})


def test_contexts_cascade_with_reingested_claims(store):
    ingest(store, SAM, DOGS)
    assert ContextStore(store.connection).get_contexts("sam-home")
    V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient({})), targets_per_window=2,
                        context_before=2).ingest(SAM)
    assert ContextStore(store.connection).get_contexts("sam-home") == {}


def test_context_store_adds_only_its_own_table(store):
    V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient({})))
    before = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ContextStore(store.connection)
    after = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert after - before == {"v2_claim_contexts"}
    columns = {row[1] for row in store.connection.execute("PRAGMA table_info(v2_claim_contexts)")}
    assert "content" not in columns  # raw message text stays in the messages table


def test_formatted_evidence_carries_memory_and_original_source(store):
    cm = ingest(store, SAM, DOGS)["m3:c1"]
    assert format_contextual_memory(cm, store) == (
        "[sam-home/mem:m3:c1] Sam's dog is Pepper. (status: current; asserted by Sam)\n"
        "source sam-home/0-2:\n"
        "[m1] user: Hello, I'm Sam.\n[m2] assistant: Nice to meet you, Sam!\n[m3] user: My dog is Pepper.")
