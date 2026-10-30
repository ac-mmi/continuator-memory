"""Query planner validation. No Ollama."""

from __future__ import annotations

import json

import pytest

from continuator_memory.generation import GenerationError
from continuator_memory.webapi.query_plan import (
    PLANNER_SYSTEM,
    analyze_intent,
    extra_content_terms,
    fallback_plan,
    is_safe_rewrite,
    plan_from_generator,
    resolve_plan,
    validate_plan,
)

QUESTION = "What fields would Priya be likely to pursue in her education?"
CHESS_Q = "When did Priya go to the chess practice group?"
RACE_Q = "When did Lena run a harbour race?"
CAMPING_Q = "When is Lena planning on going camping?"
SPEECH_Q = "When did Priya give a speech at a school?"


class ScriptedGenerator:
    model_name = "scripted-planner"

    def __init__(self, text: str | Exception):
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str, **kwargs):
        self.calls.append((system, prompt))
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def test_when_question_fallback_keeps_event_and_time():
    plan = fallback_plan(CHESS_Q)
    assert plan.queries[0] == CHESS_Q
    assert 1 <= len(plan.queries) <= 3
    assert all("chess" in query.lower() for query in plan.queries)
    assert all("practice" in query.lower() and "group" in query.lower() for query in plan.queries)
    assert all("answers this question" not in query.lower() for query in plan.queries)
    assert all("stated facts" not in query.lower() for query in plan.queries)
    assert any("date" in query.lower() or "time" in query.lower() for query in plan.queries[1:])


def test_harbour_race_and_camping_keep_event_terms():
    race = fallback_plan(RACE_Q)
    assert all("race" in query.lower() and "harbour" in query.lower() for query in race.queries)
    camping = fallback_plan(CAMPING_Q)
    assert all("camping" in query.lower() for query in camping.queries)
    assert all("stated facts" not in query.lower() for query in (*race.queries, *camping.queries))


def test_speech_at_school_is_not_routed_to_career():
    intent = analyze_intent(SPEECH_Q)
    assert intent.answer_type == "when"
    assert intent.education is False
    assert "speech" in intent.event_terms
    assert "school" in intent.event_terms
    plan = fallback_plan(SPEECH_Q)
    assert plan.queries[0] == SPEECH_Q
    joined = " ".join(plan.queries).lower()
    assert "career" not in joined
    assert "interested in" not in joined
    assert all("speech" in query.lower() and "school" in query.lower() for query in plan.queries)
    assert "time" in plan.facets or "event" in plan.facets
    assert "education" not in plan.facets


def test_fallback_includes_original_and_no_guessed_answers():
    plan = fallback_plan(QUESTION)
    assert plan.queries[0] == QUESTION
    assert 1 <= len(plan.queries) <= 3
    assert plan.entities == ("Priya",)
    assert "career" in plan.facets or "education" in plan.facets
    joined = " ".join(plan.queries).lower()
    assert "oceanography" not in joined
    assert "marine" not in joined
    for query in plan.queries:
        assert extra_content_terms(query, QUESTION) == ()


def test_rejects_generic_and_invented_rewrites():
    assert is_safe_rewrite("What did Priya say that answers this question?", CHESS_Q) is False
    assert is_safe_rewrite("Which stated facts about Priya are relevant?", CHESS_Q) is False
    assert is_safe_rewrite("When did Priya go to the chess practice group oceanography?", CHESS_Q) is False
    assert is_safe_rewrite("What career is Priya interested in?", CHESS_Q) is False
    assert is_safe_rewrite("What date did Priya go to the chess practice group?", CHESS_Q) is True


def test_validate_plan_drops_rewrites_that_change_the_need():
    plan = validate_plan(
        CHESS_Q,
        {
            "information_need": "Find when Priya went to the chess practice group",
            "entities": ["Priya"],
            "facets": ["time", "event"],
            "queries": [
                CHESS_Q,
                "What did Priya say that answers this question?",
                "What date did Priya go to the chess practice group?",
            ],
        },
    )
    assert plan.queries[0] == CHESS_Q
    assert "What did Priya say that answers this question?" not in plan.queries
    assert "What date did Priya go to the chess practice group?" in plan.queries


def test_identity_rewrite_keeps_topic_and_allows_entity_without_possessive():
    plan = fallback_plan("What is Priya's identity?")
    assert plan.queries[0] == "What is Priya's identity?"
    assert any("identity" in query.lower() for query in plan.queries)
    assert all("stated facts" not in query.lower() for query in plan.queries)
    if len(plan.queries) > 1:
        assert all("identity" in query.lower() for query in plan.queries)


def test_original_only_when_no_safe_rewrite():
    plan = validate_plan(
        CHESS_Q,
        {
            "information_need": "Find when the event happened",
            "entities": ["Priya"],
            "facets": ["time"],
            "queries": [
                CHESS_Q,
                "What did Priya say that answers this question?",
                "Which stated facts about Priya are relevant?",
            ],
        },
    )
    assert plan.queries == (CHESS_Q,)


def test_validate_plan_drops_guessed_answer_terms():
    plan = validate_plan(
        QUESTION,
        {
            "information_need": "Find Priya's likely education or work direction",
            "entities": ["Priya"],
            "facets": ["education", "career", "oceanography"],
            "queries": [
                QUESTION,
                "Priya wants to work in oceanography and marine science.",
                "What career is Priya interested in?",
                "What kind of work does Priya want to do?",
            ],
        },
    )
    assert plan.queries[0] == QUESTION
    assert all("oceanography" not in q.lower() for q in plan.queries)
    assert "oceanography" not in plan.facets
    assert "What career is Priya interested in?" in plan.queries


def test_validate_plan_requires_original_question():
    plan = validate_plan(
        QUESTION,
        {
            "information_need": "education or career evidence",
            "entity": "Priya",
            "facets": ["education"],
            "queries": [
                "What career is Priya interested in?",
                "What kind of work does Priya want to do?",
                "What does Priya plan to study or train for?",
            ],
        },
    )
    assert plan.queries[0] == QUESTION
    assert plan.entities == ("Priya",)


def test_malformed_generator_output_falls_back():
    gen = ScriptedGenerator("not json at all")
    plan = plan_from_generator(QUESTION, gen)
    assert plan.status == "planner_failed"
    assert plan.queries[0] == QUESTION
    assert gen.calls[0][0] == PLANNER_SYSTEM
    assert all("stated facts" not in query.lower() for query in plan.queries)


def test_generator_error_falls_back():
    gen = ScriptedGenerator(GenerationError("offline"))
    plan = plan_from_generator(QUESTION, gen)
    assert plan.status == "planner_failed"
    assert "offline" in (plan.error or "")
    assert plan.queries[0] == QUESTION


def test_valid_generator_json_is_accepted():
    payload = {
        "information_need": "evidence about Priya's education or work plans",
        "entities": ["Priya"],
        "facets": ["education", "career"],
        "queries": [
            QUESTION,
            "What career is Priya interested in?",
            "What kind of work does Priya want to do?",
        ],
    }
    gen = ScriptedGenerator("```json\n" + json.dumps(payload) + "\n```")
    plan = plan_from_generator(QUESTION, gen)
    assert plan.status == "ok"
    assert plan.source == "ollama"
    assert plan.queries[0] == QUESTION


def test_ollama_generic_when_rewrites_are_dropped():
    payload = {
        "information_need": "Find when Priya went to the group",
        "entities": ["Priya"],
        "facets": ["time"],
        "queries": [
            CHESS_Q,
            "What did Priya say that answers this question?",
            "Which stated facts about Priya are relevant?",
        ],
    }
    plan = plan_from_generator(CHESS_Q, ScriptedGenerator(json.dumps(payload)))
    assert plan.queries[0] == CHESS_Q
    assert all("stated facts" not in query.lower() for query in plan.queries)
    assert all("chess" in query.lower() for query in plan.queries)


def test_resolve_plan_none_uses_fallback():
    plan = resolve_plan(QUESTION, None)
    assert plan.source == "fallback"
    assert plan.queries[0] == QUESTION


def test_injected_callable_is_revalidated():
    def bad_planner(_question: str):
        return {
            "information_need": "find the answer",
            "entities": ["Priya"],
            "facets": ["career"],
            "queries": [QUESTION, "art pottery and animal welfare", "What career is Priya interested in?"],
        }

    plan = resolve_plan(QUESTION, bad_planner)
    assert all("pottery" not in q.lower() for q in plan.queries)
    assert all("animal" not in q.lower() for q in plan.queries)


def test_validate_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        validate_plan(QUESTION, ["not", "a", "plan"])
