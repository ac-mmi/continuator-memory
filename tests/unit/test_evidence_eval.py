"""Evidence evaluator validation. No Ollama."""

from __future__ import annotations

from continuator_memory.generation import GenerationError
from continuator_memory.webapi.evidence_eval import (
    EvidenceExcerpt,
    filter_followup_queries,
    heuristic_evaluation,
    resolve_evaluation,
    validate_evaluation,
)

QUESTION = "What fields would Priya be likely to pursue in her education?"
EXCERPTS = (
    EvidenceExcerpt("D1:9", "Priya: Going to continue my edu and check out career options."),
    EvidenceExcerpt(
        "D1:11",
        "Priya: I'm keen on oceanography or working in marine science.",
    ),
)


def test_followup_may_use_excerpt_terms_but_not_outside_answers():
    allowed = filter_followup_queries(
        QUESTION,
        [
            "What did Priya say about oceanography and marine science?",
            "Priya wants to work in veterinary surgery.",
            QUESTION,
        ],
        EXCERPTS,
        seen=(QUESTION,),
    )
    assert allowed == ("What did Priya say about oceanography and marine science?",)


def test_followup_rejects_terms_not_in_question_or_excerpts():
    assert filter_followup_queries(
        QUESTION,
        ["What about art pottery and animal welfare?"],
        EXCERPTS,
    ) == ()


def test_invalid_json_stops_without_inventing_queries():
    class Boom:
        def generate(self, system, prompt, **kwargs):
            return "not json"

    result = resolve_evaluation(QUESTION, EXCERPTS, Boom())
    assert result.eval_status == "evaluator_failed"
    assert result.next_queries == ()
    assert result.status == "uncertain"


def test_generator_error_stops_safely():
    class Boom:
        def generate(self, system, prompt, **kwargs):
            raise GenerationError("offline")

    result = resolve_evaluation(QUESTION, EXCERPTS, Boom())
    assert result.eval_status == "evaluator_failed"
    assert result.next_queries == ()


def test_heuristic_education_without_named_field_is_insufficient():
    excerpts = (
        EvidenceExcerpt("m-edu", "Priya: Going to continue my edu and check out career options."),
    )
    result = heuristic_evaluation(QUESTION, excerpts)
    assert result.status == "insufficient"
    assert result.missing_information
    assert result.next_queries
    joined = " ".join(result.next_queries).lower()
    assert "psycholog" not in joined
    assert "veterinary" not in joined


def test_heuristic_education_with_named_field_in_excerpts_is_sufficient():
    excerpts = (
        EvidenceExcerpt(
            "m-both",
            "Priya: I'll continue my education in oceanography or marine science.",
        ),
    )
    result = heuristic_evaluation(QUESTION, excerpts)
    assert result.status == "sufficient"
    assert result.next_queries == ()
    assert result.source == "heuristic"


def test_heuristic_followup_can_use_field_terms_from_retrieved_excerpts():
    result = heuristic_evaluation(QUESTION, EXCERPTS)
    assert result.status == "insufficient"
    joined = " ".join(result.next_queries).lower()
    assert "oceanography" in joined or "marine" in joined
    assert "psycholog" not in joined
    assert "veterinary" not in joined


def test_heuristic_when_question_requires_a_time():
    bare = (EvidenceExcerpt("m-group", "Priya: I went to the chess practice group."),)
    missing = heuristic_evaluation("When did Priya go to the chess practice group?", bare)
    assert missing.status == "insufficient"
    assert "time" in missing.missing_information or "date" in missing.missing_information
    dated = (EvidenceExcerpt("m-group", "Priya: I went to the chess practice group last May."),)
    enough = heuristic_evaluation("When did Priya go to the chess practice group?", dated)
    assert enough.status == "sufficient"
    assert enough.next_queries == ()


def test_validate_evaluation_filters_assumed_answers():
    plan = validate_evaluation(
        QUESTION,
        EXCERPTS,
        {
            "status": "insufficient",
            "supported_facts": ["Priya mentioned career options [D1:9]"],
            "missing_information": "a named field of work from the excerpts",
            "next_queries": [
                "What did Priya say about oceanography?",
                "Is she becoming a psychologist?",
            ],
        },
    )
    assert plan.status == "insufficient"
    assert plan.next_queries == ("What did Priya say about oceanography?",)
