from __future__ import annotations

from dataclasses import replace

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.rerank import CrossEncoderReranker, RerankerLoadError, rerank


def _result(rank: int, mid: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, memory_id=f"mem-{mid}", conversation_id="c", content=f"normalized {mid}",
        similarity=score, adjustment=0.0, score=score, status="active", time_class="past",
        event_time=None, valid_from=None, valid_until=None, superseded_by=None,
        sources=(MemorySource(f"mem-{mid}", f"claim-{mid}", mid, rank),), links=(),
    )


CONVERSATION = Conversation("c", tuple(
    Message(mid, "c", i, "user", text, timestamp="2023-05-20")
    for i, (mid, text) in enumerate([("m1", "I adopted a cat."), ("m2", "I moved to Lisbon."),
                                      ("m3", "Lisbon rent is high."), ("m4", "Weather chat.")])
))


class KeywordScorer:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.seen: list[tuple[str, list[str]]] = []

    def score(self, question, passages):
        self.seen.append((question, list(passages)))
        return [self.scores.get(p, 0.0) for p in passages]


def _outcome(*results):
    return RetrievalOutcome("Where did I move?", QuestionIntent(), tuple(results), (("x", "superseded"),))


def test_rerank_orders_by_cross_encoder_score_and_keeps_everything_else():
    before = _outcome(_result(1, "m1", 0.9), _result(2, "m3", 0.8), _result(3, "m2", 0.7))
    scorer = KeywordScorer({"I moved to Lisbon.": 5.0, "Lisbon rent is high.": 1.0})
    after, rows = rerank("Where did I move?", before, CONVERSATION, scorer)
    assert [r.memory_id for r in after.results] == ["mem-m2", "mem-m3", "mem-m1"]
    assert [r.rank for r in after.results] == [1, 2, 3]
    # Only the rank changes: hybrid scores, sources and the excluded list are kept.
    by_id = {r.memory_id: r for r in before.results}
    assert all(replace(r, rank=by_id[r.memory_id].rank) == by_id[r.memory_id] for r in after.results)
    assert after.excluded == before.excluded and after.question == before.question
    assert rows[0] == {"rank": 1, "memoryId": "mem-m2", "retrievalRank": 3, "retrievalScore": 0.7,
                       "rerankScore": 5.0, "sourceMessageIds": ["m2"], "passageSource": "source_message"}


def test_rerank_scores_the_original_message_text_against_the_question():
    scorer = KeywordScorer({})
    rerank("Where did I move?", _outcome(_result(1, "m2", 0.5), _result(2, "gone", 0.4)), CONVERSATION, scorer)
    # Original message text, not the memory content; the memory content only when no source resolves.
    assert scorer.seen == [("Where did I move?", ["I moved to Lisbon.", "normalized gone"])]


def test_ties_keep_the_retrieval_order():
    before = _outcome(*(_result(i, mid, 1 - i / 10) for i, mid in enumerate(["m4", "m1", "m3"], 1)))
    after, _ = rerank("q", before, CONVERSATION, KeywordScorer({}))
    assert [r.memory_id for r in after.results] == ["mem-m4", "mem-m1", "mem-m3"]


def test_empty_pool_and_wrong_score_count():
    after, rows = rerank("q", _outcome(), CONVERSATION, KeywordScorer({}))
    assert after.results == () and rows == []

    class Short:
        def score(self, question, passages):
            return [1.0]

    with pytest.raises(ValueError, match="2 candidates"):
        rerank("q", _outcome(_result(1, "m1", 0.9), _result(2, "m2", 0.8)), CONVERSATION, Short())


def test_missing_model_fails_clearly_without_downloading(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RerankerLoadError, match="(?s)not available offline.*hf download"):
        CrossEncoderReranker("continuator-test/no-such-cross-encoder")
