import ast
import inspect
import math
import random

import numpy as np
import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.v2 import memory_retrieval
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.memory import (
    LINK_CONTRADICTS,
    LINK_SUPERSEDES,
    LINK_UPDATES,
    STATUS_CONDITIONAL,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    Memory,
    MemoryLink,
    MemorySource,
)
from continuator_memory.v2.memory_retrieval import (
    DEFAULT_K,
    DEFAULT_STATE_MARGIN,
    EXCLUDED_LOW_SIMILARITY,
    EXCLUDED_SUPERSEDED,
    MemoryRetriever,
    RetrievalConfig,
    question_intent,
)
from continuator_memory.v2.memory_store import MemoryStore, consolidate_stored
from tests.helpers import HashingEmbedder
from tests.unit.test_v2_pipeline import CONVERSATION, pipeline
from tests.unit.test_v2_temporal import claim

CONV = "conv"

TIME_CLASS = {STATUS_CURRENT: "present", STATUS_SUPERSEDED: "present", STATUS_HISTORICAL: "past",
              STATUS_CONTESTED: "present", STATUS_FUTURE: "future", STATUS_CONDITIONAL: "conditional"}


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(dim=4096)
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


def memory(n, content, status=STATUS_CURRENT, superseded_by=None, conversation_id=CONV, **kw):
    return Memory(f"mem:m{n}:c1", conversation_id, content, status, f"m{n}:c1", f"m{n}", n,
                  superseded_by=superseded_by, time_class=kw.pop("time_class", TIME_CLASS[status]), **kw)


def source(mem, n, claim_n=1):
    return MemorySource(mem.memory_id, f"m{n}:c{claim_n}", f"m{n}", n, None)


def own_sources(*memories):
    return [source(m, m.position) for m in memories]


def retriever(memories, links=(), sources=None, config=None):
    r = MemoryRetriever(CountingEmbedder(), config)
    r.index(memories, own_sources(*memories) if sources is None else sources, links)
    return r


def ids(results):
    return [r.memory_id for r in results]


def contents(results):
    return [r.content for r in results]


def delhi_bangalore(delhi_text="I live in Delhi."):
    delhi = memory(1, delhi_text, STATUS_SUPERSEDED, superseded_by="mem:m2:c1")
    bangalore = memory(2, "I live in Bangalore.")
    return delhi, bangalore, [MemoryLink(bangalore.memory_id, delhi.memory_id, LINK_SUPERSEDES, "m2:c1")]


# ---- question intent ----------------------------------------------------------


@pytest.mark.parametrize("question, past, future, conditional, current", [
    ("Where do I live?", False, False, False, False),
    ("Where do I live now?", False, False, False, True),
    ("Where did I live before?", True, False, False, False),
    ("What did I used to prefer?", True, False, False, False),
    ("Where am I planning to move?", False, True, False, False),
    ("Am I considering moving to Bangalore?", False, False, True, False),
    ("Where might I move?", False, False, True, False),
    ("What happened in May?", False, False, False, False),
])
def test_question_intent_cues(question, past, future, conditional, current):
    intent = question_intent(question)
    assert (intent.past, intent.future, intent.conditional, intent.current) == (past, future, conditional, current)
    assert intent.unspecified == (not any((past, future, conditional, current)))


# ---- section 17 fixtures --------------------------------------------------------


def test_current_fact_is_retrieved():
    m1 = memory(1, "I live in Bangalore.")
    [result] = retriever([m1]).retrieve("Where do I live?")
    assert result.memory_id == m1.memory_id and result.status == STATUS_CURRENT and result.rank == 1


def test_current_version_ranks_ahead_of_superseded_version():
    delhi, bangalore, links = delhi_bangalore("I lived in Delhi.")
    outcome = retriever([delhi, bangalore], links).search("Where do I live?")
    assert ids(outcome.results)[0] == bangalore.memory_id
    assert (delhi.memory_id, EXCLUDED_SUPERSEDED) in outcome.excluded


def test_historical_question_makes_superseded_version_eligible():
    delhi, bangalore, links = delhi_bangalore()
    results = retriever([delhi, bangalore], links).retrieve("Where did I live before Bangalore?")
    assert delhi.memory_id in ids(results)
    [d] = [r for r in results if r.memory_id == delhi.memory_id]
    assert d.status == STATUS_SUPERSEDED and d.superseded_by == bangalore.memory_id and d.adjustment == 0.0


def test_future_memory_retrieved_for_planning_question():
    mumbai = memory(2, "I will move to Mumbai next year.", STATUS_FUTURE)
    results = retriever([memory(1, "I live in Bangalore."), mumbai]).retrieve("Where am I planning to move?")
    assert ids(results)[0] == mumbai.memory_id and results[0].status == STATUS_FUTURE
    assert results[0].adjustment == 0.0


def test_conditional_memory_retrieved_for_might_question():
    pune = memory(2, "If the offer works out, I may move to Pune.", STATUS_CONDITIONAL)
    results = retriever([memory(1, "I live in Bangalore."), pune]).retrieve("Where might I move?")
    assert ids(results)[0] == pune.memory_id and results[0].status == STATUS_CONDITIONAL


def test_contested_versions_both_returned_without_a_winner():
    a = memory(1, "Project launch is July 1.", STATUS_CONTESTED)
    b = memory(2, "Project launch is July 15.", STATUS_CONTESTED)
    links = [MemoryLink(b.memory_id, a.memory_id, LINK_CONTRADICTS, "m2:c1", "m2:r1")]
    results = retriever([a, b], links).retrieve("When is the project launch?")
    assert ids(results) == [a.memory_id, b.memory_id]
    assert all(r.status == STATUS_CONTESTED and r.adjustment == 0.0 for r in results)
    assert results[0].similarity == results[1].similarity
    assert [l.to_dict() for l in results[0].links] == [
        {"type": LINK_CONTRADICTS, "role": "newer", "memory_id": b.memory_id, "relation_id": "m2:r1"}]
    assert [(l.type, l.role, l.memory_id) for l in results[1].links] == [(LINK_CONTRADICTS, "older", a.memory_id)]


def test_multiple_sources_give_one_result_with_every_source():
    m = memory(1, "I live in Delhi.")
    sources = [source(m, 8, 2), source(m, 1), source(m, 12, 3)]
    results = retriever([m], sources=sources).retrieve("Where do I live?")
    assert len(results) == 1
    assert results[0].source_claim_ids == ("m1:c1", "m8:c2", "m12:c3")
    assert results[0].source_message_ids == ("m1", "m8", "m12")


def test_several_historical_versions_stay_distinct():
    olds = [memory(i, f"I lived in {city}.", STATUS_HISTORICAL) for i, city in enumerate(("Delhi", "Pune", "Chennai"), 1)]
    results = retriever(olds + [memory(4, "I live in Bangalore.")]).retrieve("Where did I live before?", k=10)
    assert set(ids(olds)) <= set(ids(results))
    assert len(ids(results)) == len(set(ids(results))) == 4


def test_irrelevant_memories_rank_below_relevant_ones():
    relevant = memory(3, "I live in Bangalore.")
    noise = [memory(1, "My favourite colour is green."), memory(2, "The dog is called Rex."),
             memory(4, "I like green tea in the morning.")]
    outcome = retriever(noise + [relevant]).search("Where do I live?")
    assert ids(outcome.results)[0] == relevant.memory_id
    assert {(m.memory_id, EXCLUDED_LOW_SIMILARITY) for m in noise[:2]} <= set(outcome.excluded)


def test_no_relevant_memory_returns_empty_result():
    noise = [memory(1, "My favourite colour is green."), memory(2, "The dog is called Rex.")]
    assert retriever(noise).retrieve("Where do I live?") == []
    assert MemoryRetriever(CountingEmbedder()).retrieve("Where do I live?") == []


def test_threshold_none_keeps_every_eligible_memory_and_higher_threshold_drops_weak_ones():
    noise = [memory(1, "My favourite colour is green."), memory(2, "I live in Bangalore.")]
    assert len(retriever(noise, config=RetrievalConfig(min_similarity=None)).retrieve("Where do I live?")) == 2
    assert retriever(noise, config=RetrievalConfig(min_similarity=0.9)).retrieve("Where do I live?") == []


def test_equal_scores_have_stable_order():
    ms = [memory(3, "I like tea."), memory(1, "I like tea."), memory(2, "I like tea.")]
    first = retriever(ms).retrieve("Do I like tea?")
    assert ids(first) == ["mem:m1:c1", "mem:m2:c1", "mem:m3:c1"]
    for seed in range(5):
        shuffled = ms[:]
        random.Random(seed).shuffle(shuffled)
        assert [r.to_dict() for r in retriever(shuffled).retrieve("Do I like tea?")] == [r.to_dict() for r in first]


def test_tie_prefers_current_status_before_position():
    old = memory(1, "I like tea.", STATUS_HISTORICAL)
    new = memory(2, "I like tea.")
    config = RetrievalConfig(state_margin=0.0)
    assert ids(retriever([new, old], config=config).retrieve("Do I like tea?")) == [new.memory_id, old.memory_id]
    assert ids(retriever([new, old], config=config).retrieve("Did I used to like tea?")) == [
        old.memory_id, new.memory_id]


# ---- section 18: temporal retrieval ---------------------------------------------


def slot_fixture():
    delhi, bangalore, links = delhi_bangalore()
    mumbai = memory(3, "I will live in Mumbai next year.", STATUS_FUTURE)
    pune = memory(4, "If the offer works out, I may live in Pune.", STATUS_CONDITIONAL)
    chennai = memory(5, "I used to live in Chennai.", STATUS_HISTORICAL)
    return [delhi, bangalore, mumbai, pune, chennai], links


def test_current_question_prefers_current_and_demotes_other_states():
    ms, links = slot_fixture()
    outcome = retriever(ms, links).search("Where do I live now?", k=10)
    assert outcome.results[0].content == "I live in Bangalore."
    assert "I live in Delhi." not in contents(outcome.results)
    for r in outcome.results[1:]:
        assert r.status != STATUS_CURRENT and r.adjustment == -DEFAULT_STATE_MARGIN


def test_historical_question_does_not_suppress_past_versions():
    ms, links = slot_fixture()
    results = retriever(ms, links).retrieve("Where did I live before?", k=10)
    by_content = {r.content: r for r in results}
    assert {"I live in Delhi.", "I used to live in Chennai."} <= set(by_content)
    assert by_content["I live in Delhi."].adjustment == by_content["I used to live in Chennai."].adjustment == 0.0
    assert by_content["I will live in Mumbai next year."].adjustment == -DEFAULT_STATE_MARGIN


def test_future_question_aligns_future_and_conditional():
    ms, links = slot_fixture()
    results = retriever(ms, links).retrieve("Where will I live next year?", k=10)
    assert results[0].content == "I will live in Mumbai next year."
    assert {r.status: r.adjustment for r in results}[STATUS_CONDITIONAL] == 0.0


def test_conditional_question_retrieves_conditional_and_future():
    ms, links = slot_fixture()
    results = retriever(ms, links).retrieve("Am I considering living in Pune?", k=10)
    adjustments = {r.status: r.adjustment for r in results}
    assert adjustments[STATUS_CONDITIONAL] == adjustments[STATUS_FUTURE] == 0.0
    assert adjustments[STATUS_HISTORICAL] == -DEFAULT_STATE_MARGIN
    ranked = [r.status for r in results]
    assert ranked.index(STATUS_CONDITIONAL) < ranked.index(STATUS_HISTORICAL)
    assert "If the offer works out, I may live in Pune." in contents(results[:2])


def test_ambiguous_question_keeps_candidates_and_falls_back_to_similarity():
    ms, links = slot_fixture()
    outcome = retriever(ms, links).search("Tell me about Mumbai.", k=10)
    assert outcome.intent.unspecified
    assert outcome.results[0].content == "I will live in Mumbai next year."
    assert outcome.results[0].status == STATUS_FUTURE


def test_superseded_without_indexed_replacement_stays_eligible():
    delhi = memory(1, "I live in Delhi.", STATUS_SUPERSEDED, superseded_by="mem:m9:c1")
    [result] = retriever([delhi]).retrieve("Where do I live?")
    assert result.memory_id == delhi.memory_id and result.adjustment == -DEFAULT_STATE_MARGIN


def test_superseded_excluded_via_update_link_even_without_superseded_by():
    delhi = memory(1, "I live in Delhi.", STATUS_SUPERSEDED)
    bangalore = memory(2, "I live in Bangalore.")
    links = [MemoryLink(bangalore.memory_id, delhi.memory_id, LINK_UPDATES, "m2:c1", "m2:r1")]
    assert ids(retriever([delhi, bangalore], links).retrieve("Where do I live?")) == [bangalore.memory_id]


def test_future_and_current_same_slot():
    current = memory(1, "I live in Bangalore.")
    future = memory(2, "I will live in Bangalore next year.", STATUS_FUTURE)
    r = retriever([current, future])
    assert ids(r.retrieve("Do I live in Bangalore now?")) == [current.memory_id, future.memory_id]
    assert ids(r.retrieve("Will I live in Bangalore next year?"))[0] == future.memory_id


def test_status_margin_breaks_equal_similarity_only():
    hist = memory(1, "I live in Delhi.", STATUS_HISTORICAL, time_class="ambiguous_past")
    cur = memory(2, "I live in Delhi.")
    [first, second] = retriever([hist, cur]).retrieve("Where do I live?")
    assert (first.memory_id, second.memory_id) == (cur.memory_id, hist.memory_id)
    assert first.similarity == second.similarity
    assert second.score == pytest.approx(second.similarity - DEFAULT_STATE_MARGIN)


# ---- top-K ------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_top_k(k):
    ms = [memory(i, f"I visited city number {i} on a trip.") for i in range(1, 13)]
    r = retriever(ms)
    assert len(r.retrieve("Which city did I visit on a trip?", k=k)) == k
    assert [x.rank for x in r.retrieve("Which city did I visit on a trip?", k=k)] == list(range(1, k + 1))


def test_default_k_and_configured_k():
    ms = [memory(i, f"I visited city number {i} on a trip.") for i in range(1, 13)]
    assert DEFAULT_K == 5 and len(retriever(ms).retrieve("Which city did I visit?")) == 5
    assert len(retriever(ms, config=RetrievalConfig(k=3)).retrieve("Which city did I visit?")) == 3
    with pytest.raises(ValueError):
        RetrievalConfig(k=0)
    with pytest.raises(ValueError):
        retriever(ms).retrieve("Which city?", k=0)


# ---- indexing lifecycle -----------------------------------------------------------


def test_reindexing_same_memories_creates_no_duplicates_or_embeddings():
    delhi, bangalore, links = delhi_bangalore()
    r = retriever([delhi, bangalore], links)
    calls = len(r.embedder.embedded)
    stats = r.index([delhi, bangalore], own_sources(delhi, bangalore), links)
    assert (stats.added, stats.updated, stats.unchanged, stats.removed) == (0, 0, 2, 0)
    assert len(r) == 2 and len(r.embedder.embedded) == calls


def test_changed_content_is_reembedded():
    m = memory(1, "I live in Delhi.")
    r = retriever([m])
    changed = memory(1, "I own a red bicycle.")
    stats = r.index([changed], own_sources(changed))
    assert stats.updated == 1 and r.embedder.embedded[-1] == "I own a red bicycle."
    assert r.retrieve("Where is Delhi?") == []
    assert contents(r.retrieve("Do I own a bicycle?")) == ["I own a red bicycle."]


def test_memories_missing_from_snapshot_are_dropped_not_fabricated():
    a, b = memory(1, "I live in Delhi."), memory(2, "I live in Bangalore.")
    r = retriever([a, b])
    assert r.index([b], own_sources(b)).removed == 1
    assert ids(r.retrieve("Where do I live?")) == [b.memory_id]
    assert r.index([], conversation_id=CONV).removed == 1 and len(r) == 0


def test_snapshot_sync_is_per_conversation():
    a = memory(1, "I live in Delhi.", conversation_id="one")
    b = memory(1, "I live in Delhi.", conversation_id="two")
    r = MemoryRetriever(CountingEmbedder())
    r.index([a])
    r.index([b])
    assert len(r) == 2
    assert [x.conversation_id for x in r.retrieve("Where do I live?")] == ["one", "two"]
    assert [x.conversation_id for x in r.retrieve("Where do I live?", conversation_id="two")] == ["two"]


def test_superseded_memories_remain_indexed():
    delhi, bangalore, links = delhi_bangalore()
    r = retriever([delhi, bangalore], links)
    assert len(r) == 2
    assert delhi.memory_id in ids(r.retrieve("Where did I live before?"))


# ---- provenance and diagnostics ---------------------------------------------------


def test_result_exposes_full_provenance_and_temporal_fields():
    m = memory(1, "I moved to Bangalore in March 2024.", valid_from="2024-03", event_time=None,
               time_class="transition")
    [r] = retriever([m]).retrieve("When did I move to Bangalore?")
    d = r.to_dict()
    for key in ("memory_id", "content", "similarity", "adjustment", "score", "status", "time_class", "event_time",
                "valid_from", "valid_until", "sources", "links", "superseded_by"):
        assert key in d
    assert (d["valid_from"], d["time_class"]) == ("2024-03", "transition")
    assert d["sources"] == [{"memory_id": m.memory_id, "claim_id": "m1:c1", "message_id": "m1", "position": 1,
                             "observed_at": None}]


def test_debug_representation():
    delhi, bangalore, links = delhi_bangalore()
    outcome = retriever([delhi, bangalore], links).search("Where do I live now?")
    debug = outcome.debug()
    assert debug["intent"]["current"] is True and debug["intent"]["cues"] == ["now"]
    assert debug["excluded"] == [{"memory_id": delhi.memory_id, "reason": EXCLUDED_SUPERSEDED}]
    [row] = debug["results"]
    assert set(row) == {"rank", "memory_id", "status", "similarity", "adjustment", "score", "source_claim_ids",
                        "source_message_ids", "links"}
    assert row["links"] == [f"{LINK_SUPERSEDES} older {delhi.memory_id}"]


def test_identical_inputs_give_identical_output():
    ms, links = slot_fixture()
    runs = [retriever(ms, links).search("Where did I live before?", k=10).debug() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# ---- end to end over real consolidation -------------------------------------------


def test_retrieval_over_consolidated_claims():
    c = consolidate(CONV, [
        claim("m1:c1", "The user lives in Delhi.", 1, claim_type="residence"),
        claim("m2:c1", "The user now lives in Bangalore.", 2, claim_type="residence"),
        claim("m3:c1", "The user will move to Mumbai next year.", 3, claim_type="residence"),
    ])
    by_content = {m.content: m for m in c.memories.values()}
    assert by_content["The user lives in Delhi."].status == STATUS_SUPERSEDED
    r = MemoryRetriever(CountingEmbedder())
    r.index_consolidator(c)
    assert contents(r.retrieve("Where does the user live?"))[0] == "The user now lives in Bangalore."
    assert "The user lives in Delhi." not in contents(r.retrieve("Where does the user live?"))
    assert "The user lives in Delhi." in contents(r.retrieve("Where did the user live before?"))
    assert contents(r.retrieve("Where is the user planning to move?"))[0] == "The user will move to Mumbai next year."


def test_retrieval_from_memory_store_traces_to_raw_message(store):
    p, _ = pipeline(store)
    p.ingest(parse_conversation(CONVERSATION))
    consolidate_stored(store.connection, "trip")
    r = MemoryRetriever(CountingEmbedder())
    r.index_store(MemoryStore(store.connection), "trip")
    now = r.retrieve("Where does Leah live now?")
    assert now[0].content == "Leah actually lives in Valencia now."
    assert "Leah lives in Madrid." not in contents(now)
    before = r.retrieve("Where did Leah live before?")
    [madrid] = [x for x in before if x.content == "Leah lives in Madrid."]
    assert madrid.source_claim_ids == ("m1:c2",)
    assert [(l.type, l.role, l.memory_id) for l in madrid.links] == [(LINK_UPDATES, "newer", "mem:m4:c1")]
    raw = store.get_messages("trip", list(madrid.source_message_ids))[0]
    assert raw.content == "My sister Leah lives in Madrid."


# ---- isolation ----------------------------------------------------------------------


def test_module_does_no_raw_retrieval_or_generation():
    tree = ast.parse(inspect.getsource(memory_retrieval))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and ("generation" in m or "ollama" in m or m.endswith(".store") or "pipeline" in m)
                   for m in imported)
    source_text = inspect.getsource(memory_retrieval)
    assert "load_embeddings" not in source_text and "get_messages" not in source_text


# ---- experimarine contextual retrieval ------------------------------------------------

PET_QUESTION = "What pet does Priya have?"
ASKING_TURN = "Priya: That's so nice! What pet do you have?"
ANSWER_TURN = "Priya: Yup, I do- Biscuit, my tortoise."
ANSWER_CONTEXT = "Lena: Do you have pets?\n" + ANSWER_TURN


class ScriptedEmbedder:
    """Known unit vectors so cosines are exact; records every embedded text."""

    def __init__(self, cosines: dict[str, float]):
        self.vectors = {text: np.array([c, math.sqrt(1.0 - c * c)]) for text, c in cosines.items()}
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return np.stack([self.vectors[text] for text in texts])


def pet_embedder(**extra):
    return ScriptedEmbedder({PET_QUESTION: 1.0, ASKING_TURN: 0.8, ANSWER_TURN: 0.7, ANSWER_CONTEXT: 0.95, **extra})


def pet_retriever(config, contexts=None, embedder=None):
    asking, answer = memory(1, ASKING_TURN), memory(2, ANSWER_TURN)
    r = MemoryRetriever(embedder or pet_embedder(), config)
    r.index([asking, answer], own_sources(asking, answer), contexts=contexts)
    return r, asking, answer


def pet_contexts():
    return {(CONV, memory(2, ANSWER_TURN).memory_id): ANSWER_CONTEXT}


def test_context_weight_zero_matches_indexing_without_context():
    plain, _, _ = pet_retriever(RetrievalConfig(k=2))
    with_context, _, _ = pet_retriever(RetrievalConfig(k=2), pet_contexts())
    expected = [r.to_dict() for r in plain.retrieve(PET_QUESTION)]
    assert [r.to_dict() for r in with_context.retrieve(PET_QUESTION)] == expected
    assert all("context_similarity" not in row for row in expected)
    assert [row["content"] for row in expected] == [ASKING_TURN, ANSWER_TURN]


def test_no_contexts_means_no_context_embedding():
    embedder = pet_embedder()
    pet_retriever(RetrievalConfig(k=2, context_weight=0.5), embedder=embedder)
    assert ANSWER_CONTEXT not in embedder.embedded


def test_context_promotes_answer_turn_but_returns_the_memory_itself():
    r, asking, answer = pet_retriever(RetrievalConfig(k=2, context_weight=0.5), pet_contexts())
    first, second = r.retrieve(PET_QUESTION)
    assert (first.memory_id, second.memory_id) == (answer.memory_id, asking.memory_id)
    assert first.content == ANSWER_TURN and first.source_message_ids == (answer.message_id,)
    assert first.similarity == pytest.approx(0.7)
    assert first.context_similarity == pytest.approx(0.95)
    assert first.score == pytest.approx(0.5 * 0.7 + 0.5 * 0.95)
    assert first.to_dict()["context_similarity"] == pytest.approx(0.95)
    assert second.context_similarity is None and second.score == pytest.approx(0.8)


def test_context_blend_combines_with_adjustment_and_lexical():
    config = RetrievalConfig(k=2, context_weight=0.5, lexical_weight=0.25)
    r, _, answer = pet_retriever(config, pet_contexts())
    [row] = [x for x in r.retrieve(PET_QUESTION) if x.memory_id == answer.memory_id]
    lexical = memory_retrieval.lexical_overlap(PET_QUESTION, ANSWER_TURN)
    assert row.score == pytest.approx(0.5 * 0.7 + 0.5 * 0.95 + row.adjustment + 0.25 * lexical)


def test_reindexing_same_contexts_embeds_nothing_and_changed_context_is_reembedded():
    embedder = pet_embedder(**{"Lena: Any pets?\n" + ANSWER_TURN: 0.9})
    r, asking, answer = pet_retriever(RetrievalConfig(k=2, context_weight=0.5), pet_contexts(), embedder=embedder)
    calls = len(embedder.embedded)
    stats = r.index([asking, answer], own_sources(asking, answer), contexts=pet_contexts())
    assert (stats.added, stats.updated, stats.unchanged, stats.removed) == (0, 0, 2, 0)
    assert len(embedder.embedded) == calls
    changed = {(CONV, answer.memory_id): "Lena: Any pets?\n" + ANSWER_TURN}
    stats = r.index([asking, answer], own_sources(asking, answer), contexts=changed)
    assert (stats.added, stats.updated, stats.unchanged) == (0, 0, 2)
    assert embedder.embedded[calls:] == ["Lena: Any pets?\n" + ANSWER_TURN]
    [row] = [x for x in r.retrieve(PET_QUESTION) if x.memory_id == answer.memory_id]
    assert row.context_similarity == pytest.approx(0.9)


def test_context_weight_must_be_between_zero_and_one():
    assert RetrievalConfig().context_weight == 0.0
    with pytest.raises(ValueError, match="context_weight"):
        RetrievalConfig(context_weight=-0.1)
    with pytest.raises(ValueError, match="context_weight"):
        RetrievalConfig(context_weight=1.1)
