import json
import sqlite3

import numpy as np
import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.context import SpeakerRegistry, contextualize_claims, windows_by_message
from continuator_memory.v2.context_store import ContextStore, contextualize_stored
from continuator_memory.v2.entities import (
    NAME_ALIAS,
    NAME_CANDIDATE,
    NAME_CANONICAL,
    ORGANIZATION,
    OTHER,
    PERSON,
    PET,
    PRODUCT,
    PROJECT,
    REF_AMBIGUOUS,
    ROLE_MENTIONED,
    ROLE_POSSESSOR,
    ROLE_SPEAKER,
    ROLE_SUBJECT,
    CueEntityTyper,
    entity_evidence,
    entity_id,
    resolve_entities,
)
from continuator_memory.v2.entity_store import EntityStore, resolve_entities_stored, stored_entity_memories
from continuator_memory.v2.memory_store import MemoryStore, consolidate_stored
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.windowing import build_windows


class ClaimsClient:
    """Scripted atomizer: (content, entities) claims per target message."""

    model_name = "scripted-atomizer"

    def __init__(self, claims):
        self.claims = claims

    def complete_json(self, system, prompt, schema, examples=()):
        ids = schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"]
        entries = []
        for mid in ids:
            claims = [{"claim_id": f"{mid}:c{i}", "content": content, "entities": list(entities), "qualifiers": [],
                       "temporal": {"expression": "", "resolved_date": ""}, "polarity": "positive",
                       "claim_type": "", "source_message_id": mid}
                      for i, (content, *rest) in enumerate(self.claims.get(mid, ()), 1)
                      for entities in [rest[0] if rest else ()]]
            entries.append({"source_message_id": mid, "status": "claims" if claims else "no_durable_assertion",
                            "claims": claims, "explicit_relations": []})
        return json.dumps({"messages": entries})


def conversation(cid, *messages):
    return parse_conversation({"conversation_id": cid, "messages": [
        {"id": f"m{i}", "role": role, "content": content} for i, (role, content) in enumerate(messages, 1)]})


def build(store, conv, claims):
    report = V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient(claims)), targets_per_window=2,
                                 context_before=2).ingest(conv)
    assert not report.issues, report.issues
    cid = conv.conversation_id
    consolidate_stored(store.connection, cid)
    contextualize_stored(store.connection, cid)
    resolve_entities_stored(store.connection, cid)
    return EntityStore(store.connection).get(cid), {m.memory.claim_id: m for m in
                                                    stored_entity_memories(store.connection, cid)}


def refs(memory, role):
    return [r.entity_id for r in memory.entity_refs if r.role == role]


def by_name(resolution, name):
    return next(e for e in resolution.entities.values() if e.canonical_name == name)


ONE_CONV = conversation("pets", ("user", "Hello, I'm Sam."), ("assistant", "Hello Sam, I'm Alex."),
                        ("user", "My dog is Pepper."), ("assistant", "My dog is Biscuit."))
ONE_CONV_CLAIMS = {"m3": [("The user's dog is Pepper.", ["Pepper"])],
                   "m4": [("The assistant's dog is Biscuit.", ["Biscuit"])]}


# ---- 1. Sam vs Alex ---------------------------------------------------------


def test_sam_and_alex_in_one_conversation_are_distinct_entities(store):
    res, mems = build(store, ONE_CONV, ONE_CONV_CLAIMS)
    sam, alex = by_name(res, "Sam"), by_name(res, "Alex")
    assert sam.entity_id != alex.entity_id
    assert (sam.speaker_role, sam.entity_type, alex.speaker_role) == ("user", PERSON, "assistant")
    assert refs(mems["m3:c1"], ROLE_POSSESSOR) == [sam.entity_id]
    assert refs(mems["m4:c1"], ROLE_POSSESSOR) == [alex.entity_id]
    assert sam.entity_id not in mems["m4:c1"].entity_ids() and alex.entity_id not in mems["m3:c1"].entity_ids()
    pepper = by_name(res, "Pepper")
    assert pepper.entity_type == PET and refs(mems["m3:c1"], ROLE_MENTIONED) == [pepper.entity_id]


def test_sam_and_alex_in_two_conversations_like_phase4(store):
    sam_res, sam = build(store, conversation("sam-home", ("user", "Hello, I'm Sam."), ("user", "My dog is Pepper.")),
                         {"m2": [("The user's dog is Pepper.", ["Pepper"])]})
    alex_res, alex = build(store, conversation("alex-home", ("user", "Hello, I'm Alex."), ("user", "My dog is Biscuit.")),
                           {"m2": [("The user's dog is Biscuit.", ["Biscuit"])]})
    assert sam["m2:c1"].memory.memory_id == alex["m2:c1"].memory.memory_id  # local IDs collide
    assert refs(sam["m2:c1"], ROLE_POSSESSOR) == [by_name(sam_res, "Sam").entity_id] == ["ent:sam-home/role:user"]
    assert refs(alex["m2:c1"], ROLE_POSSESSOR) == [by_name(alex_res, "Alex").entity_id] == ["ent:alex-home/role:user"]


# ---- 2 / 8. same name across conversations, isolation -----------------------


def test_same_name_in_two_conversations_is_not_merged(store):
    a_res, a = build(store, conversation("a", ("user", "Hi, I'm Sam."), ("user", "My dog is Pepper.")),
                     {"m2": [("The user's dog is Pepper.", ["Pepper"])]})
    b_res, b = build(store, conversation("b", ("user", "Hi, I'm Sam."), ("user", "My dog is Biscuit.")),
                     {"m2": [("The user's dog is Biscuit.", ["Biscuit"])]})
    sam_a, sam_b = by_name(a_res, "Sam"), by_name(b_res, "Sam")
    assert sam_a.entity_id != sam_b.entity_id and sam_a.conversation_id != sam_b.conversation_id
    assert a_res.memories_of(sam_b.entity_id) == [] and b_res.memories_of(sam_a.entity_id) == []
    assert "Biscuit" not in {e.canonical_name for e in a_res.entities.values()}


def test_identical_third_party_names_stay_conversation_scoped(store):
    claims = {"m1": [("Leah lives in Madrid.", ["Leah", "Madrid"])]}
    a_res, _ = build(store, conversation("trip-a", ("user", "Leah lives in Madrid.")), claims)
    b_res, _ = build(store, conversation("trip-b", ("user", "Leah lives in Madrid.")), claims)
    assert by_name(a_res, "Leah").entity_id == "ent:trip-a/name:leah"
    assert by_name(b_res, "Leah").entity_id == "ent:trip-b/name:leah"
    rows = store.connection.execute("SELECT conversation_id, entity_id FROM v2_entities WHERE local_key = "
                                    "'name:leah' ORDER BY conversation_id").fetchall()
    assert rows == [("trip-a", "ent:trip-a/name:leah"), ("trip-b", "ent:trip-b/name:leah")]


def test_entity_id_is_injective_for_separator_characters():
    pairs = [("a/b", "name:c"), ("a", "b/name:c"), ("a:b", "name:c"), ("a", "b:name:c")]
    assert len({entity_id(*p) for p in pairs}) == len(pairs)


# ---- 3 / 4. aliases ---------------------------------------------------------


ALIAS_CONV = conversation("alias", ("user", "My name is Samuel."), ("user", "Call me Sam."),
                          ("assistant", "Sam, your cat Miso sounds lovely."))


def test_explicit_alias_joins_the_speaker_entity(store):
    res, mems = build(store, ALIAS_CONV, {"m3": [("Sam's cat Miso sounds lovely.", ["Sam", "Miso"])]})
    user = res.entities["ent:alias/role:user"]
    assert (user.canonical_name, user.aliases) == ("Samuel", ("Sam",))
    assert [(n.name, n.kind, n.message_id) for n in user.names] == [("Samuel", NAME_CANONICAL, "m1"),
                                                                      ("Sam", NAME_ALIAS, "m2")]
    assert refs(mems["m3:c1"], ROLE_POSSESSOR) == [user.entity_id]
    assert "ent:alias/name:sam" not in res.entities


def test_speaker_registry_records_call_me_as_alias_of_the_stated_name():
    sp = SpeakerRegistry(ALIAS_CONV.messages).speaker("user", 2)
    assert (sp.name, sp.status, sp.evidence, sp.aliases) == ("Samuel", "named", ("m1",), (("Sam", "m2"),))
    assert sp.to_dict()["aliases"] == [["Sam", "m2"]]
    plain = SpeakerRegistry(conversation("p", ("user", "Hi, I'm Sam.")).messages).speaker("user", 0)
    assert "aliases" not in plain.to_dict()  # Phase 5A serialization unchanged


def test_similar_names_without_a_declaration_are_not_merged(store):
    conv = conversation("fuzzy", ("user", "Hi, I'm Samuel."), ("user", "My friend Sam lives in Porto."))
    res, mems = build(store, conv, {"m2": [("Sam lives in Porto.", ["Sam", "Porto"])]})
    user, sam = res.entities["ent:fuzzy/role:user"], res.entities["ent:fuzzy/name:sam"]
    assert user.canonical_name == "Samuel" and user.aliases == () and sam.canonical_name == "Sam"
    assert refs(mems["m2:c1"], ROLE_SUBJECT) == [sam.entity_id]


def test_two_third_party_spellings_are_two_entities(store):
    conv = conversation("names", ("user", "Sam lives in Porto."), ("user", "Samuel lives in Braga."))
    res, _ = build(store, conv, {"m1": [("Sam lives in Porto.", ["Sam", "Porto"])],
                                 "m2": [("Samuel lives in Braga.", ["Samuel", "Braga"])]})
    assert {"ent:names/name:sam", "ent:names/name:samuel"} <= set(res.entities)


# ---- 5. subject vs mentioned ------------------------------------------------


def test_subject_and_mentioned_entities_are_distinguished(store):
    conv = conversation("work", ("user", "Hi, I'm Priya."), ("user", "My colleague Sam works at OpenAI."),
                        ("user", "Sam bought a MacBook from Apple."))
    res, mems = build(store, conv, {"m2": [("Sam is the user's colleague.", ["Sam"]),
                                           ("Sam works at OpenAI.", ["Sam", "OpenAI"])],
                                    "m3": [("Sam bought a MacBook from Apple.", ["Sam", "MacBook", "Apple"])]})
    sam, openai = res.entities["ent:work/name:sam"], res.entities["ent:work/name:openai"]
    work = mems["m2:c2"]
    assert refs(work, ROLE_SUBJECT) == [sam.entity_id] and refs(work, ROLE_MENTIONED) == [openai.entity_id]
    assert refs(work, ROLE_SPEAKER) == ["ent:work/role:user"]
    assert (sam.entity_type, openai.entity_type) == (PERSON, ORGANIZATION)
    mac = mems["m3:c1"]
    assert refs(mac, ROLE_SUBJECT) == [sam.entity_id]
    assert refs(mac, ROLE_MENTIONED) == ["ent:work/name:macbook", "ent:work/name:apple"]
    assert res.entities["ent:work/name:macbook"].entity_type == PRODUCT
    assert res.entities["ent:work/name:apple"].entity_type == OTHER  # no explicit cue: not guessed
    colleague = mems["m2:c1"]
    assert refs(colleague, ROLE_SUBJECT) == [sam.entity_id]
    assert refs(colleague, ROLE_MENTIONED) == ["ent:work/role:user"]  # "the user" mentioned, not subject


def test_lowercase_things_are_not_entities(store):
    conv = conversation("picnic", ("user", "The picnic will be in Hillside Park."))
    res, _ = build(store, conv, {"m1": [("The picnic will be in Hillside Park.", ["picnic", "Hillside Park"])]})
    assert set(res.entities) == {"ent:picnic/role:user", "ent:picnic/name:hillside park"}


def test_project_entity_type(store):
    conv = conversation("atlas", ("user", "Project Atlas launches on June 3rd."))
    res, _ = build(store, conv, {"m1": [("Project Atlas launches on June 3rd.", ["Project Atlas"])]})
    assert res.entities["ent:atlas/name:project atlas"].entity_type == PROJECT


def test_cue_typer_conflicting_cues_give_other():
    texts = ["The user works at Jordan.", "Jordan is the user's brother."]
    assert CueEntityTyper().type_of(["Jordan"], texts) == OTHER
    assert CueEntityTyper().type_of(["Jordan"], texts[1:]) == PERSON


# ---- speakers: unnamed, ambiguous, assistant --------------------------------


def test_unnamed_user_is_a_known_actor_without_an_invented_name(store):
    res, mems = build(store, conversation("anon", ("user", "My cat is Miso.")),
                      {"m1": [("The user's cat is Miso.", ["Miso"])]})
    user = res.entities["ent:anon/role:user"]
    assert (user.canonical_name, user.status, user.names, user.label) == (None, "unnamed", (), "the user")
    assert refs(mems["m1:c1"], ROLE_POSSESSOR) == [user.entity_id]


def test_conflicting_speaker_names_stay_ambiguous(store):
    conv = conversation("shared", ("user", "Hi, I'm Sam."), ("user", "My name is Jordan, I share this account."),
                        ("user", "Sam lives in Porto."))
    res, mems = build(store, conv, {"m3": [("Sam lives in Porto.", ["Sam", "Porto"])]})
    user = res.entities["ent:shared/role:user"]
    assert (user.canonical_name, user.status) == (None, "ambiguous")
    assert [(n.name, n.kind) for n in user.names] == [("Sam", NAME_CANDIDATE), ("Jordan", NAME_CANDIDATE)]
    assert refs(mems["m3:c1"], ROLE_SUBJECT) == ["ent:shared/name:sam"]  # not joined to the ambiguous user


def test_a_name_shared_by_two_speakers_is_an_ambiguous_reference(store):
    conv = conversation("twins", ("user", "Hi, I'm Sam."), ("assistant", "Hello! My name is Sam as well."),
                        ("user", "Sam lives in Porto."))
    res, mems = build(store, conv, {"m3": [("Sam lives in Porto.", ["Sam", "Porto"])]})
    subject = [r for r in mems["m3:c1"].entity_refs if r.role == ROLE_SUBJECT]
    assert [(r.entity_id, r.status) for r in subject] == [(None, REF_AMBIGUOUS)]
    assert "identity_ambiguous" in subject[0].detail and "ent:twins/name:sam" not in res.entities


def test_assistant_statements_reference_the_assistant_not_the_user(store):
    conv = conversation("advice", ("user", "Hi, I'm Priya."), ("assistant", "Lisbon has excellent public transport."))
    res, mems = build(store, conv, {"m2": [("Lisbon has excellent public transport.", ["Lisbon"])]})
    m = mems["m2:c1"]
    assert refs(m, ROLE_SPEAKER) == ["ent:advice/role:assistant"]
    assert "ent:advice/role:user" not in m.entity_ids() and "ent:advice/role:user" not in res.entities


# ---- phase 2C explicit identity metadata ------------------------------------


def _claims_path(conv, claims):
    contexts = contextualize_claims(conv.messages, claims, windows_by_message(build_windows(conv, 2, 2)))
    c = consolidate(conv.conversation_id, claims)
    return resolve_entities(conv.conversation_id, conv.messages, claims, contexts, c.memories.values(), c.sources)


def test_phase2c_entity_ids_and_aliases_are_reused():
    conv = conversation("ids", ("user", "Sam moved to Pune."), ("user", "Samuel likes chess."),
                        ("user", "Sammy plays tennis."))
    p = [Provenance.of(m) for m in conv.messages]
    claims = [Claim("m1:c1", "Sam moved to Pune.", p[0], ("Sam", "Pune"), metadata={"entity_ids": {"Sam": "p1"}}),
              Claim("m2:c1", "Samuel likes chess.", p[1], ("Samuel",), metadata={"entity_ids": {"Samuel": "p1"}}),
              Claim("m3:c1", "Sammy plays tennis.", p[2], ("Sammy",), metadata={"aliases": {"Sammy": "Sam"}})]
    res = _claims_path(conv, claims)
    person = res.entities["ent:ids/id:p1"]
    assert person.canonical_name == "Sam" and person.aliases == ("Samuel", "Sammy")
    assert res.memories_of(person.entity_id, ROLE_SUBJECT) == ["mem:m1:c1", "mem:m2:c1", "mem:m3:c1"]


def test_phase2c_alias_ambiguity_stays_explicit():
    conv = conversation("amb", ("user", "Sam moved."), ("user", "Sam cooks."), ("user", "Kit sings."))
    p = [Provenance.of(m) for m in conv.messages]
    claims = [Claim("m1:c1", "Sam moved.", p[0], ("Sam",), metadata={"aliases": {"Kit": "Sam"}}),
              Claim("m2:c1", "Sam cooks.", p[1], ("Sam",), metadata={"aliases": {"Kit": "Kim"}}),
              Claim("m3:c1", "Kit sings.", p[2], ("Kit",))]
    res = _claims_path(conv, claims)
    kit = [r for r in res.refs["mem:m3:c1"] if r.role != ROLE_SPEAKER]
    assert [(r.role, r.entity_id, r.status) for r in kit] == [(ROLE_SUBJECT, None, REF_AMBIGUOUS)]
    assert "ent:amb/name:kit" not in res.entities


# ---- 6. provenance ----------------------------------------------------------


def test_every_entity_resolves_to_its_source(store):
    res, _ = build(store, ALIAS_CONV, {"m3": [("Sam's cat Miso sounds lovely.", ["Sam", "Miso"])]})
    contexts = ContextStore(store.connection).get_contexts("alias")
    assert res.entities
    for e in res.entities.values():
        evidence = entity_evidence(e, contexts, store)
        assert evidence[0].messages[-1].message_id == e.established_by_message_id
        texts = {m.message_id: m.content for ctx in evidence for m in ctx.messages}
        for n in e.names:
            assert n.name in texts[n.message_id]
    user = res.entities["ent:alias/role:user"]
    windows = [ctx.window.message_ids for ctx in entity_evidence(user, contexts, store)]
    assert ("m1",) in windows and ("m2",) in windows


# ---- 7 / 9. re-ingestion and determinism ------------------------------------


def test_reingestion_does_not_duplicate_entities(store):
    first, _ = build(store, ONE_CONV, ONE_CONV_CLAIMS)
    second, _ = build(store, ONE_CONV, ONE_CONV_CLAIMS)
    assert first.to_dict() == second.to_dict()
    count = store.connection.execute("SELECT COUNT(*), COUNT(DISTINCT entity_id) FROM v2_entities").fetchone()
    assert count == (len(first.entities), len(first.entities))


def test_reingested_claims_cascade_entities_away(store):
    build(store, ONE_CONV, ONE_CONV_CLAIMS)
    V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient({}))).ingest(ONE_CONV)
    for table in ("v2_entities", "v2_entity_aliases", "v2_memory_entities"):
        assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_v1_conversation_replacement_cascades(store):
    build(store, ONE_CONV, ONE_CONV_CLAIMS)
    store.save_conversation(ONE_CONV, [], np.zeros((0, 0), dtype=np.float32), "none")
    assert store.connection.execute("SELECT COUNT(*) FROM v2_entities").fetchone() == (0,)


def test_two_independent_runs_are_byte_identical():
    def snapshot():
        s = Store(":memory:")
        try:
            out = []
            for conv, claims in ((ONE_CONV, ONE_CONV_CLAIMS),
                                 (ALIAS_CONV, {"m3": [("Sam's cat Miso sounds lovely.", ["Sam", "Miso"])]})):
                res, mems = build(s, conv, claims)
                out.append([res.to_dict(), [m.to_dict() for m in mems.values()]])
            rows = [s.connection.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3").fetchall()
                    for t in ("v2_entities", "v2_entity_aliases", "v2_memory_entities")]
            return json.dumps([out, rows])
        finally:
            s.close()

    assert snapshot() == snapshot()


def test_stored_and_in_memory_resolution_match(store):
    stored, _ = build(store, ONE_CONV, ONE_CONV_CLAIMS)
    claims = SemanticStore(store.connection).get_claims("pets")
    assert _claims_path(ONE_CONV, claims).to_dict() == stored.to_dict()


# ---- storage ----------------------------------------------------------------


def test_entity_store_adds_only_its_tables_and_enforces_ambiguity(store):
    V2IngestionPipeline(store, SemanticAtomizer(ClaimsClient({})))
    ContextStore(store.connection)
    MemoryStore(store.connection)
    before = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    EntityStore(store.connection)
    after = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert after - before == {"v2_entities", "v2_entity_aliases", "v2_memory_entities"}
    build(store, ONE_CONV, ONE_CONV_CLAIMS)
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            "INSERT INTO v2_memory_entities VALUES ('pets', 'mem:m3:c1', 99, 'mentioned', NULL, 'x', 'm3:c1', 'm3', "
            "'resolved', NULL)")
