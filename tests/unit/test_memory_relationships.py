import numpy as np
import pytest

from continuator_memory.memory.consolidation import decide_relationship
from continuator_memory.memory.extraction import MemoryExtractor
from continuator_memory.memory.formation import MemoryFormation
from continuator_memory.memory.models import Action, Relationship
from continuator_memory.memory.store import MemoryStore
from continuator_memory.models import parse_conversation
from tests.helpers import ScriptedJSONClient, doc, mem, rel

TEXTS = [
    "Marco Silva leads Project Nova.",
    "Priya handles the budget for Nova.",
    "Priya is from Oslo and joined Project Nova in May.",
    "Hana Kobayashi takes over leading Nova from Marco.",
    "Marco still leads the Orion project.",
]
SCRIPT = {
    TEXTS[0]: doc(relationships=[rel("Marco Silva", "leads", "Project Nova")]),
    TEXTS[1]: doc(relationships=[rel("Priya", "handles", "budget"),
                                 rel("Priya", "works_on", "Nova")]),
    TEXTS[2]: doc([mem("Priya", "from", "Oslo")],
                  [rel("Priya", "works_on", "Project Nova", valid_from="May")]),
    TEXTS[3]: doc(relationships=[rel("Hana Kobayashi", "leads", "Nova", "update", "Marco")]),
    TEXTS[4]: doc(relationships=[rel("Marco", "leads", "Orion project")]),
}


@pytest.fixture
def ms(store):
    conv = parse_conversation(
        {"conversation_id": "c1",
         "messages": [{"id": f"m{i + 1}", "role": "user", "content": t} for i, t in enumerate(TEXTS)]}
    )
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    memory = MemoryStore(store.connection)
    MemoryFormation(memory, MemoryExtractor(ScriptedJSONClient(SCRIPT)), clock=lambda: "T").form(conv)
    return memory


def _entity(ms, name):
    return ms.find_entity("c1", name)


def test_relationship_creation(ms):
    first = ms.relationships("c1")[0]
    assert (first.subject, first.predicate, first.object) == ("Marco Silva", "leads", "Project Nova")
    assert first.source_message_ids == ("m1",)
    assert first.subject_entity_id != first.object_entity_id


def test_aliases_resolve_to_one_entity(ms):
    names = sorted(e.name for e in ms.entities("c1"))
    assert names == ["Hana Kobayashi", "Marco Silva", "Orion project", "Priya", "Project Nova",
                     "budget"]


def test_relationship_lookup(ms):
    priya = _entity(ms, "Priya")
    handles = ms.relationships("c1", entity_id=priya.entity_id, predicate="handles")
    assert [(r.subject, r.object) for r in handles] == [("Priya", "budget")]
    works_on = ms.relationships("c1", predicate="works_on")
    assert len(works_on) == 1 and works_on[0].source_message_ids == ("m2", "m3")


def test_relationship_traversal(ms):
    budget = _entity(ms, "budget")
    ((edge, other_id),) = ms.neighbors(budget.entity_id, current_only=True)
    assert edge.predicate == "handles"
    home = ms.memories("c1", entity_id=other_id, predicate="from")
    assert [m.value for m in home] == ["Oslo"]


def test_multiple_relationships_for_one_entity(ms):
    nova = _entity(ms, "Nova")
    edges = {(r.subject, r.predicate, r.is_current) for r in ms.relationships(entity_id=nova.entity_id)}
    assert edges == {
        ("Marco Silva", "leads", False),
        ("Hana Kobayashi", "leads", True),
        ("Priya", "works_on", True),
    }


def test_historical_relationship_supersession(ms):
    leads = {(r.subject, r.object): r for r in ms.relationships("c1", predicate="leads")}
    marco_nova = leads[("Marco Silva", "Project Nova")]
    hana_nova = leads[("Hana Kobayashi", "Project Nova")]
    assert marco_nova.is_current is False
    assert hana_nova.is_current is True
    assert hana_nova.supersedes_relationship_id == marco_nova.relationship_id
    # Marco leading another project is a separate edge stated later: still current.
    assert leads[("Marco Silva", "Orion project")].is_current is True


def _r(rid, subj, obj, current=True, sources=("m1",)):
    return Relationship(rid, "c", subj, subj, "p", obj, obj, "state", sources, 0, "T", None, None,
                        current, None)


@pytest.mark.parametrize(
    "subj, obj, kind, previous, slot, action",
    [
        ("a", "b", "state", None, [], Action.CREATE),
        ("a", "b", "state", None, [_r("x", "a", "b")], Action.UPDATE),
        ("a", "b", "state", None, [_r("x", "a", "b", sources=("m9",))], Action.KEEP),
        ("a", "c", "state", None, [_r("x", "a", "b")], Action.CREATE),
        ("a", "c", "update", None, [_r("x", "a", "b")], Action.SUPERSEDE),
        ("a", "c", "reference", None, [_r("x", "a", "b")], Action.IGNORE),
        ("a", "b", "state", None, [_r("x", "a", "b", current=False)], Action.IGNORE),
    ],
)
def test_decide_relationship(subj, obj, kind, previous, slot, action):
    assert decide_relationship(subj, obj, kind, previous, slot, "m9").action is action


def test_relationship_update_naming_previous_narrows_targets():
    slot = [_r("x", "marco", "nova"), _r("y", "hana", "orion")]
    d = decide_relationship("hana", "nova", "update", "marco", slot, "m9")
    assert d.supersede_ids == ("x",)
