"""Experimarine graph retrieval index (HippoRAG-style PPR over turns and phrases)."""

from __future__ import annotations

import numpy as np
import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.graph_index import GraphIndex, phrases, split_speaker
from tests.helpers import HashingEmbedder


def _msgs(rows, cid="c"):
    """rows: (message_id, content, timestamp)."""
    return [Message(mid, cid, i, "user", text, ts) for i, (mid, text, ts) in enumerate(rows)]


def _graph(rows, **kw):
    g = GraphIndex(HashingEmbedder(), **kw)
    g.add_messages(_msgs(rows))
    return g


# ---- multi-hop --------------------------------------------------------------------------------

MULTI = [
    ("m0", "Lena: I signed up for a pottery class last week.", "2023-05-01T10:00:00"),
    ("m1", "Priya: Nice, enjoy it!", "2023-05-01T10:00:00"),
    ("m2", "Priya: The weather was rainy today.", "2023-06-01T10:00:00"),
    ("m3", "Lena: Pottery workshops are held in Brooklyn.", "2023-07-01T10:00:00"),
    ("m4", "Priya: I bought new running shoes.", "2023-08-01T10:00:00"),
]
QUESTION = "Which city hosts the class Lena signed up for?"


def test_two_hop_turn_without_question_words_is_reached_through_a_shared_phrase():
    g = _graph(MULTI, adjacency=False)
    out = g.search(QUESTION)
    ids = [h.message_id for h in out.hits]
    body = split_speaker(MULTI[3][1], "user")[1]
    assert not (phrases(body).keys() & phrases(QUESTION).keys())  # the indexed text shares no question phrase
    assert ids[0] == "m0"
    assert "m3" in ids
    hop = {h.message_id: h.hops for h in out.hits}
    assert hop["m0"] == 1 and hop["m3"] == 3  # seed phrase -> m0 -> "pottery" -> m3
    assert "m2" not in ids and "m4" not in ids  # unconnected turns get no mass


def test_scores_form_a_probability_ranking_and_ignore_other_conversations():
    g = GraphIndex(HashingEmbedder(), adjacency=False)
    g.add_messages(_msgs(MULTI, "c"))
    g.add_messages(_msgs([("x0", "Lena: pottery class in Paris", "2023-01-01T00:00:00")], "other"))
    hits = g.search(QUESTION, conversation_id="c").hits
    assert all(h.conversation_id == "c" for h in hits)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


# ---- source attribution ---------------------------------------------------------------------

def test_hits_and_mentions_keep_source_turn_speaker_and_timestamp():
    g = _graph(MULTI, adjacency=False)
    hit = next(h for h in g.search(QUESTION).hits if h.message_id == "m3")
    assert (hit.speaker, hit.timestamp) == ("Lena", "2023-07-01T10:00:00")
    mention = g.observations("pottery")
    assert [(m.message_id, m.speaker, m.timestamp) for m in mention] == [
        ("m0", "Lena", "2023-05-01T10:00:00"), ("m3", "Lena", "2023-07-01T10:00:00")]
    assert split_speaker("Priya: hi there", "user") == ("Priya", "hi there")
    assert split_speaker("plain text", "assistant") == ("assistant", "plain text")
    assert "priya" not in g.phrase_index  # speaker labels are attributes, not phrase nodes


# ---- temporal updates -----------------------------------------------------------------------

UPDATE = [
    ("s1", "I have a violin lesson with Dr. Okafor every two weeks.", "2023/04/03 (Mon) 14:18"),
    ("s2", "I made pasta for dinner.", "2023/06/01 (Thu) 10:00"),
    ("s3", "Speaking of scales, I see Dr. Okafor every week now.", "2023/11/03 (Fri) 19:56"),
]


def test_updates_keep_every_dated_statement_and_none_is_marked_current():
    g = _graph(UPDATE, adjacency=False)
    obs = g.observations("Okafor")
    assert [(m.message_id, m.timestamp) for m in obs] == [("s1", "2023/04/03 (Mon) 14:18"),
                                                         ("s3", "2023/11/03 (Fri) 19:56")]
    hits = {h.message_id: h for h in g.search("How often do I see Dr. Okafor?").hits}
    assert {"s1", "s3"} <= set(hits)  # both versions are returned, each with its own date
    assert hits["s1"].timestamp != hits["s3"].timestamp
    assert not hasattr(hits["s3"], "current") and "current" not in g.stats()


def test_incremarine_add_appends_without_rewriting_and_matches_a_full_build():
    full = _graph(UPDATE, adjacency=False)
    inc = GraphIndex(HashingEmbedder(), adjacency=False)
    inc.add_messages(_msgs(UPDATE[:2]))
    before = list(inc.edges)
    report = inc.add_messages(_msgs(UPDATE))  # re-adding indexed turns is a no-op
    assert report["turns"] == 1 and inc.edges[: len(before)] == before
    a = {h.message_id: h.score for h in full.search("How often do I see Dr. Okafor?").hits}
    b = {h.message_id: h.score for h in inc.search("How often do I see Dr. Okafor?").hits}
    assert a.keys() == b.keys() and all(np.isclose(a[k], b[k]) for k in a)


# ---- duplicate evidence ---------------------------------------------------------------------

def test_duplicate_turns_stay_separate_and_repeated_words_count_once():
    rows = [("d1", "Rafael: The studio opens in June.", "2023-01-01T00:00:00"),
            ("d2", "Rafael: The studio opens in June.", "2023-02-01T00:00:00"),
            ("d3", "Noor: studio studio studio", "2023-03-01T00:00:00")]
    g = _graph(rows, adjacency=False)
    hits = g.search("When does the studio open?").hits
    ids = [h.message_id for h in hits]
    assert sorted(ids) == ["d1", "d2", "d3"] and len(ids) == len(set(ids))
    assert {h.timestamp for h in hits if h.message_id in ("d1", "d2")} == {"2023-01-01T00:00:00",
                                                                          "2023-02-01T00:00:00"}
    studio_edges = [e for e in g.edges if e[5] == "mention" and e[3] == g.phrase_index["studio"]]
    assert len(studio_edges) == 3  # one per turn, not one per occurrence


# ---- missing graph links --------------------------------------------------------------------

def test_question_with_no_known_phrase_returns_nothing_and_reports_unmatched():
    g = _graph(MULTI, adjacency=False)
    out = g.search("Zebras?")
    assert out.hits == () and out.unmatched == ("zebra",)


def test_turn_without_phrases_is_only_reachable_through_adjacency():
    rows = [("a0", "Rafael: Did you finish the marathon training plan?", "2023-01-01T00:00:00"),
            ("a1", "Noor: Yes!", "2023-01-01T00:00:00")]
    assert "a1" not in [h.message_id for h in _graph(rows, adjacency=False).search("marathon training").hits]
    assert "a1" in [h.message_id for h in _graph(rows, adjacency=True).search("marathon training").hits]


def test_empty_index_is_safe():
    assert GraphIndex(HashingEmbedder()).search("anything").hits == ()


# ---- harness integration ----------------------------------------------------------------------

@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from continuator_memory.webapi.locomo_report import build_memory_retriever, retrieval_config
    from continuator_memory.webapi.workspace import Workspace

    monkeypatch.setenv("CONTINUATOR_LEXICAL_WEIGHT", "0.25")
    monkeypatch.setenv("CONTINUATOR_CONTEXT_WEIGHT", "0.5")
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=5)
    ws.import_text("\n\n".join(f"You: I visited museum number {i} with my sister" for i in range(20)),
                   filename="c.txt", extract="messages", db_path=tmp_path / "w.db")
    ws._eval_retriever = build_memory_retriever(ws.conversation, ws.claims, ws.relations, ws.embedder(),
                                                retrieval_config("hybrid"))
    return ws


def _item(ws):
    from continuator_memory.webapi.locomo_report import LocomoQuestion, classify_evidence
    gold = [ws.conversation.messages[7].message_id]
    return LocomoQuestion(0, "Which museum did I visit with my sister?", "museum 7", 1, gold,
                          classify_evidence(gold, {m.message_id for m in ws.conversation.messages}))


def test_graph_modes_are_switchable_and_leave_the_baseline_untouched(workspace):
    from continuator_memory.webapi.locomo_report import evaluate_question

    before = evaluate_question(_item(workspace), workspace, mode="hybrid", top_k=5, depth=20, generate=False)
    graph = evaluate_question(_item(workspace), workspace, mode="graph", top_k=5, depth=20, generate=False)
    fused = evaluate_question(_item(workspace), workspace, mode="graph_hybrid", top_k=5, depth=20, generate=False)
    after = evaluate_question(_item(workspace), workspace, mode="hybrid", top_k=5, depth=20, generate=False)
    assert before["error"] is None and graph["error"] is None and fused["error"] is None
    assert before["rankedMessageIds"] == after["rankedMessageIds"]  # baseline unchanged by graph use
    assert before["retrievalTrace"] is None
    assert graph["retrievalTrace"]["mode"] == "graph" and graph["retrievalTrace"]["seeds"]
    assert len(set(graph["rankedMessageIds"])) == len(graph["rankedMessageIds"])
    assert set(fused["rankedMessageIds"]) >= set(before["rankedMessageIds"][:5]) - set()
    ids = {m.message_id for m in workspace.conversation.messages}
    assert set(graph["promptMessageIds"]) <= ids  # evidence is original turns
