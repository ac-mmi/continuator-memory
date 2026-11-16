import json

import pytest

from continuator_memory.store import Store
from continuator_memory.v2 import evidence_diagnostic as ed
from continuator_memory.v2.atomizer import SemanticAtomizer
from continuator_memory.v2.expansion_store import build_knowledge_stored
from continuator_memory.v2.pipeline import V2IngestionPipeline
from tests.unit.test_v2_expansion import ScriptedClient, conversation, user

CASES = {
    "pets": ([("user", "Hi, my name is Sam. I have a dog called Pepper."),
              ("assistant", "Nice to meet you. The vet on Rua Nova opens at 9am."),
              ("user", "I start my new job on the 2nd of February."),
              ("user", "I don't eat meat.")],
             {"m1": [("The user has a dog called Pepper.", ["Pepper"])],
              "m2": [("The vet on Rua Nova opens at 9am.", ["Rua Nova"])],
              "m3": [("The user starts a new job.", [])],
              "m4": [("The user eats meat.", [])]}, {}),
    "team": (user("Jonas is our accountant.", "Mei and Jonas will book the train tickets.",
                  "The budget is €20,000.", "Actually, the budget is now €25,000.", "Mei likes jazz.",
                  "Our office is in Porto."),
             {"m1": [("Jonas is an accountant.", ["Jonas"])],
              "m2": [("Mei will book the train tickets.", ["Mei"])],
              "m3": [("The budget is €20,000.", ["budget"])],
              "m4": [("The budget is now €25,000.", ["budget"])],
              "m5": [("Mei likes jazz.", ["Mei"])],
              "m6": [("The office is in Porto.", ["Porto"])]},
             {"m4": [("m4:c1", "m3:c1", "UPDATES", "Actually")]}),
}


@pytest.fixture(scope="module")
def views():
    store = Store(":memory:")
    out = {}
    for cid, (messages, claims, relations) in CASES.items():
        report = V2IngestionPipeline(store, SemanticAtomizer(ScriptedClient(claims, relations)), targets_per_window=2,
                                     context_before=2).ingest(conversation(cid, *messages))
        assert not report.issues, report.issues
        out[cid] = ed.load_view(store.connection, cid, build_knowledge_stored(store.connection, cid))
    return out


def compare(views, cid, message_id, memory_ids=None):
    return ed.compare_message(views[cid], message_id, memory_ids)


# ---- provenance and source mapping --------------------------------------------------------


def test_message_to_memory_mapping_and_provenance(views):
    v = views["team"]
    ids = [f"m{i}" for i in range(1, 7)]
    assert v.memories_of == {m: (f"mem:{m}:c1",) for m in ids}
    assert {m: v.source_messages(m) for m in v.knowledge.memories} == {f"mem:{m}:c1": (m,) for m in ids}
    assert all(ed.provenance(v, m) == {"conversation": True, "source_message": True, "source_window": True,
                                       "claim": True, "complete": True} for m in v.knowledge.memories)


def test_provenance_detects_a_missing_claim(views):
    v = views["team"]
    broken = ed.ConversationView(v.knowledge, {k: c for k, c in v.claims.items() if k != "m2:c1"}, v.sources)
    assert ed.provenance(broken, "mem:m2:c1")["claim"] is False
    assert ed.compare_message(broken, "m2").loss_flags()[-1] == ed.PROVENANCE_LOSS


def test_source_windows_end_at_the_claim_message(views):
    v = views["team"]
    assert v.windows("mem:m4:c1") == (v.message_window("m4"),)
    assert v.window_messages("mem:m4:c1") == {"m1", "m2", "m3", "m4"}
    assert v.window_messages("mem:m6:c1") == {"m3", "m4", "m5", "m6"}
    assert ed.message_item(v, "m4", 1, None, ed.ORIGIN_HYBRID).source_window_ids == (v.message_window("m4"),)


# ---- information-preservation classification ----------------------------------------------


def test_preservation_states():
    assert ed._compare(set(), {"x"}).state == ed.NOT_APPLICABLE
    assert ed._compare({"x"}, {"x", "y"}).state == ed.PRESERVED
    assert ed._compare({"x"}, set()).state == ed.LOST
    assert ed._compare({"x"}, set(), ambiguous=["ambiguous_past"]).state == ed.AMBIGUOUS


def test_only_claim_bearing_sentences_are_compared(views):
    assert views["pets"].claim_text("m2") == "The vet on Rua Nova opens at 9am."
    assert views["pets"].claim_text("m1") == "I have a dog called Pepper."


# ---- speaker --------------------------------------------------------------------------------


def test_generic_user_label_is_recorded_and_phase5a_names_the_speaker(views):
    c = compare(views, "pets", "m1")
    assert (c.speaker, c.contextual_speaker) == (ed.SPEAKER_ROLE, ed.SPEAKER_NAMED)
    assert ed.SPEAKER_LOSS not in c.loss_flags()


def test_assistant_assertion_without_attribution_is_missing(views):
    c = compare(views, "pets", "m2")
    assert (c.speaker, c.contextual_speaker) == (ed.SPEAKER_MISSING, ed.SPEAKER_ROLE)
    assert c.loss_flags() == [ed.SPEAKER_LOSS]


def test_speaker_attribution_variants(views):
    v = views["pets"]
    assert ed.speaker_attribution(v, "mem:m1:c1", "Sam has a dog called Pepper.") == ed.SPEAKER_NAMED
    assert ed.speaker_attribution(v, "mem:m1:c1", "Sam's dog is called Pepper.") == ed.SPEAKER_NAMED
    assert ed.speaker_attribution(v, "mem:m1:c1", "The assistant has a dog called Pepper.") == ed.SPEAKER_DIFFERENT
    assert ed.speaker_attribution(v, "mem:m1:c1", "A dog is called Pepper.") == ed.SPEAKER_MISSING
    assert ed.speaker_attribution(views["team"], "mem:m1:c1", "Jonas is an accountant.") == ed.SPEAKER_NA


# ---- entity / temporal / polarity -----------------------------------------------------------


def test_entity_written_in_the_source_but_in_no_memory_is_lost(views):
    c = compare(views, "team", "m2")
    assert (c.entity.state, c.entity.lost, c.entity.represented) == (
        ed.LOST, ("ent:team/name:jonas",), ("ent:team/name:mei",))
    assert compare(views, "pets", "m1").entity.state == ed.PRESERVED


def test_temporal_markers_are_compared_with_the_memory(views):
    lost = compare(views, "pets", "m3").temporal
    assert (lost.state, lost.lost) == (ed.LOST, ("date:2nd of february",))
    kept = compare(views, "pets", "m2").temporal
    assert (kept.state, kept.represented) == (ed.PRESERVED, ("date:9am",))
    assert compare(views, "team", "m2").temporal.represented == ("future",)


def test_polarity_and_correction(views):
    assert compare(views, "pets", "m4").polarity.lost == ("negation",)
    corrected = compare(views, "team", "m4").polarity
    assert (corrected.state, corrected.represented) == (ed.PRESERVED, ("correction",))


def test_context_comparison_uses_only_the_given_memories(views):
    c = compare(views, "team", "m2", memory_ids=[])
    assert c.memory_ids == () and c.entity.state == ed.LOST and c.speaker == ed.SPEAKER_NA


# ---- fragmentation --------------------------------------------------------------------------


def test_links_and_group_categories(views):
    team = [("team", m) for m in ("mem:m3:c1", "mem:m4:c1")]
    assert ed.memory_links(views, *team) == (ed.LINK_EDGE, ed.LINK_WINDOW)
    assert ed.group_category(views, team) == ed.MULTI_MEMORY_RELATED
    mei = [("team", "mem:m2:c1"), ("team", "mem:m5:c1")]
    assert ed.memory_links(views, *mei) == (ed.LINK_SAME_ENTITY,)
    assert ed.group_category(views, mei) == ed.MULTI_MEMORY_SAME_ENTITY
    apart = [("team", "mem:m1:c1"), ("team", "mem:m6:c1")]
    assert ed.memory_links(views, *apart) == () and ed.group_category(views, apart) == ed.MULTI_MEMORY_DISCONNECTED
    assert ed.group_category(views, [("team", "mem:m1:c1")]) == ed.SINGLE_MEMORY_EVIDENCE
    assert ed.memory_links(views, ("team", "mem:m1:c1"), ("pets", "mem:m1:c1")) == ()
    assert ed.components(views, apart) == [[apart[0]], [apart[1]]]
    assert ed.components(views, [*apart, *mei]) == [[*apart, *mei]]


# ---- items and metrics ----------------------------------------------------------------------


def test_items_are_deterministic_and_carry_provenance(views):
    v = views["pets"]
    first = [ed.memory_item(v, m, i, 0.5).to_dict() for i, m in enumerate(v.knowledge.memories, 1)]
    second = [ed.memory_item(v, m, i, 0.5).to_dict() for i, m in enumerate(v.knowledge.memories, 1)]
    assert json.dumps(first) == json.dumps(second)
    item = ed.memory_item(v, "mem:m1:c1", 1, 0.5)
    assert (item.global_memory_id, item.message_ids, item.claim_ids) == ("pets/mem:m1:c1", ("m1",), ("m1:c1",))
    assert item.speaker["attribution"] == ed.SPEAKER_ROLE and item.speaker["contextual_attribution"] == ed.SPEAKER_NAMED
    raw = ed.message_item(v, "m1", 1, None, ed.ORIGIN_RAW)
    assert raw.memory_ids == ("mem:m1:c1",) and raw.speaker["name"] == "Sam"


def test_assembly_metrics(views):
    v = views["team"]
    items = [ed.memory_item(v, m, i, None) for i, m in enumerate(("mem:m3:c1", "mem:m4:c1", "mem:m1:c1"), 1)]
    m = ed.assembly_metrics(views, items)
    assert (m["items"], m["unique_source_messages"], m["relationships"], m["duplicate_message_rate"]) == (3, 3, 1, 0.0)
    assert ed.distribution([1, 2.0, None, 1]) == {"1": 2, "2.0": 1, "none": 1}


# ---- no leakage -----------------------------------------------------------------------------


def test_module_never_reads_a_question_or_gold_label():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ed))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {n for n in names if any(w in n.lower() for w in ("question", "gold", "expected", "answer"))}
