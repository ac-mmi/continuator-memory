"""Prompt construction and answer serialization. Mocked generators only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from continuator_memory.generation import GenerationError
from continuator_memory.models import Conversation, Message
from continuator_memory.webapi.answering import (
    NOT_CONFIGURED_ANSWER,
    SYSTEM_PROMPT,
    PromptExcerpt,
    build_answer_context,
    context_to_evidence_fields,
    excerpt_supports_claim,
    generate_answer,
    parse_generation_overrides,
    load_generation_settings,
    GenerationSettings,
    validate_answer_citations,
)
from continuator_memory.webapi.sample import sample_conversation
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

QUESTION = "What does Alex commute on?"


def _evidence(*, selected: list[str], dropped: list[str] | None = None, matches: list):
    return SimpleNamespace(
        matches=matches,
        selection=SimpleNamespace(selected=tuple(selected), dropped=tuple(dropped or ())),
    )


def _match(memory_id: str, statement: str, sources: list[str], *, rank=1, score=0.9, similarity=0.9):
    return SimpleNamespace(
        memory_id=memory_id,
        statement=statement,
        rank=rank,
        score=score,
        similarity=similarity,
        source_message_ids=tuple(sources),
    )


def test_prompt_cites_source_and_memory_ids_not_hidden_retrieval():
    conversation = sample_conversation()
    evidence = _evidence(
        selected=["m-commute"],
        dropped=["m-guitar"],
        matches=[
            _match("mem:seed", "Alex commutes on a weekday bicycle.", ["m-commute"]),
            _match("mem:gold", "Alex owns a Fender Stratocaster.", ["m-guitar"], rank=2, score=0.4),
        ],
    )
    ctx = build_answer_context(
        QUESTION,
        evidence,
        conversation,
        seed_ids=["mem:seed"],
        expanded_ids=["mem:gold"],
        max_context_chars=12_000,
    )
    assert "[m-commute]" in ctx.prompt
    assert "mem:seed" in ctx.prompt
    assert "m-guitar" not in ctx.prompt
    assert ctx.retrieved_memory_ids == ("mem:seed", "mem:gold")
    assert ctx.prompt_memory_ids == ("mem:seed",)
    assert ctx.dropped_memory_ids == ("mem:gold",)
    assert ctx.dropped_message_ids == ("m-guitar",)
    by_id = {row["id"]: row for row in ctx.memories}
    assert by_id["mem:seed"]["inPrompt"] is True
    assert by_id["mem:gold"]["inPrompt"] is False
    assert by_id["mem:gold"]["origin"] == "bprime"
    serialized = context_to_evidence_fields(ctx)
    assert serialized["promptMemoryIds"] == ["mem:seed"]
    assert serialized["droppedMemoryIds"] == ["mem:gold"]
    assert serialized["question"] == QUESTION
    commute = next(e for e in ctx.excerpts if e.message_id == "m-commute")
    dated = f"(message date: {commute.message_date}) " if commute.message_date else ""
    assert f"[{commute.message_id}] {dated}{commute.content}" in ctx.prompt
    assert "(role:" not in ctx.prompt
    assert "linked memory mem:seed" in ctx.prompt


def test_system_prompt_requires_supporting_excerpt_and_forbids_invented_ids():
    lower = SYSTEM_PROMPT.lower()
    assert "do not invent" in lower
    assert "source ids" in lower or "source id" in lower
    assert "same line" in lower
    assert "degrees" in lower and "certifications" in lower


def test_generation_truncation_drops_later_excerpts_from_the_prompt():
    conversation = sample_conversation()
    commute = next(m.content for m in conversation.messages if m.message_id == "m-commute")
    evidence = _evidence(
        selected=["m-commute", "m-guitar"],
        matches=[
            _match("mem:seed", "commute", ["m-commute"]),
            _match("mem:gold", "guitar", ["m-guitar"], rank=2),
        ],
    )
    ctx = build_answer_context(
        QUESTION,
        evidence,
        conversation,
        seed_ids=["mem:seed"],
        expanded_ids=["mem:gold"],
        max_context_chars=max(20, len(commute)),
    )
    assert ctx.truncated is True
    assert "m-commute" in {e.message_id for e in ctx.excerpts}
    assert "m-guitar" not in {e.message_id for e in ctx.excerpts}
    assert "mem:gold" in ctx.dropped_memory_ids
    assert ctx.memories[1]["inPrompt"] is False


def test_mocked_generate_answer_keeps_provenance_and_settings():
    conversation = sample_conversation()
    evidence = _evidence(
        selected=["m-commute"],
        matches=[_match("mem:seed", "commute", ["m-commute"])],
    )
    ctx = build_answer_context(
        QUESTION, evidence, conversation, seed_ids=["mem:seed"], expanded_ids=[], max_context_chars=4000,
    )

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, *, temperature=None, num_predict=None):
            assert system == SYSTEM_PROMPT
            assert QUESTION in prompt
            assert "[m-commute]" in prompt
            self.kwargs = {"temperature": temperature, "num_predict": num_predict}
            return "Alex commutes on a weekday bicycle [m-commute]."

    fake = Fake()
    settings = GenerationSettings(enabled=True, model="fake-reader", temperature=0.2, num_predict=64)
    text, meta = generate_answer(ctx, settings, fake)
    assert text == "Alex commutes on a weekday bicycle [m-commute]."
    assert meta["status"] == "ok"
    assert meta["model"] == "fake-reader"
    assert meta["provider"] == "ollama"
    assert meta["temperature"] == 0.2
    assert "host" not in meta
    assert "apiKey" not in meta
    assert fake.kwargs == {"temperature": 0.2, "num_predict": 64}


def test_not_configured_does_not_fabricate_an_answer():
    conversation = sample_conversation()
    evidence = _evidence(selected=["m-commute"], matches=[_match("mem:seed", "c", ["m-commute"])])
    ctx = build_answer_context(
        QUESTION, evidence, conversation, seed_ids=["mem:seed"], expanded_ids=[], max_context_chars=4000,
    )
    settings = GenerationSettings(enabled=True, model="")
    text, meta = generate_answer(ctx, settings, None)
    assert text == NOT_CONFIGURED_ANSWER
    assert meta["status"] == "not_configured"
    assert meta["configured"] is False


def test_provider_error_is_surfaced():
    conversation = sample_conversation()
    evidence = _evidence(selected=["m-commute"], matches=[_match("mem:seed", "c", ["m-commute"])])
    ctx = build_answer_context(
        QUESTION, evidence, conversation, seed_ids=["mem:seed"], expanded_ids=[], max_context_chars=4000,
    )

    class Boom:
        model_name = "x"

        def generate(self, system, prompt):
            raise GenerationError("could not reach Ollama")

    text, meta = generate_answer(ctx, GenerationSettings(enabled=True, model="x"), Boom())
    assert "could not reach Ollama" in text
    assert meta["status"] == "error"


def test_parse_generation_rejects_secrets_and_hosts():
    base = GenerationSettings(enabled=True, model="qwen2.5:1.5b")
    with pytest.raises(ValueError, match="credentials"):
        parse_generation_overrides({"apiKey": "secret"}, base)
    with pytest.raises(ValueError, match="host"):
        parse_generation_overrides({"host": "http://evil"}, base)
    with pytest.raises(ValueError, match="model"):
        parse_generation_overrides({"model": "http://evil/model"}, base)
    out = parse_generation_overrides({"temperature": 0.5, "numPredict": 32, "maxContextChars": 800}, base)
    assert out.temperature == 0.5
    assert out.num_predict == 32
    assert out.max_context_chars == 800


def test_workspace_mocked_answer_retains_ids_and_distinguishes_prompt():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt):
            return "Alex commutes on a weekday bicycle [m-commute]."

    workspace.set_generator(Fake())
    result = workspace.ask(
        "weekday bicycle",
        retrieval_mode="baseline",
        generate=True,
        generation={"model": "fake-reader", "maxContextChars": 4000},
    )
    evidence = result["answer"]["evidence"]
    assert result["answer"]["content"] == "Alex commutes on a weekday bicycle [m-commute]."
    assert evidence["question"] == "weekday bicycle"
    assert evidence["memoryIds"]
    assert evidence["promptMemoryIds"]
    assert set(evidence["promptMemoryIds"]) <= set(evidence["memoryIds"])
    assert "m-commute" in evidence["selectedMessageIds"]
    assert evidence["generation"]["status"] == "ok"
    assert evidence["generation"]["model"] == "fake-reader"
    assert "host" not in evidence["generation"]
    in_prompt = {row["id"]: row["inPrompt"] for row in evidence["memories"]}
    for mid in evidence["promptMemoryIds"]:
        assert in_prompt[mid] is True
    for mid in evidence["droppedMemoryIds"]:
        assert in_prompt[mid] is False


def test_workspace_bprime_still_expands_when_generating():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    from continuator_memory.v2.memory import memory_id

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt):
            return "grounded"

    workspace.set_generator(Fake())
    result = workspace.ask(
        "weekday bicycle",
        retrieval_mode="bprime",
        generate=True,
        generation={"model": "fake-reader"},
    )
    evidence = result["answer"]["evidence"]
    gold = memory_id("gold")
    assert gold in evidence["expandedMemoryIds"]
    assert gold in evidence["memoryIds"]
    assert evidence["hops"][0]["pairId"] == "P01"
    assert result["answer"]["content"] == "grounded"


def _education_conversation() -> Conversation:
    rows = (
        ("D1:9", "Priya: Going to continue my edu and check out career options."),
        ("D1:11", "Priya: I'm keen on oceanography or working in marine science."),
        ("D10:15", "Priya: Cool! What did it look like?"),
    )
    messages = tuple(
        Message(mid, "imported", index, "user", text, None)
        for index, (mid, text) in enumerate(rows)
    )
    return Conversation("imported", messages, title="imported")


def _education_context():
    conversation = _education_conversation()
    evidence = _evidence(
        selected=["D1:9", "D1:11", "D10:15"],
        matches=[
            _match("mem:d1-9", "education", ["D1:9"], rank=1),
            _match("mem:d1-11", "oceanography", ["D1:11"], rank=2),
            _match("mem:d10-15", "look", ["D10:15"], rank=3, score=0.3),
        ],
    )
    return build_answer_context(
        "What fields would Priya be likely to pursue in her education?",
        evidence,
        conversation,
        seed_ids=["mem:d1-9"],
        expanded_ids=[],
        max_context_chars=12_000,
    )


def test_excerpt_support_rejects_same_topic_excerpt_missing_timing_fact():
    question = "When did Priya go to the chess practice group?"
    claim = (
        "Priya went to the chess practice group yesterday according to her "
        "statement in message D1"
    )
    d1_3 = "Priya: I went to a chess practice group yesterday and it was so useful."
    d1_7 = (
        "Priya: The practice group has taught me patience and given me "
        "confidence to enter a tournament."
    )
    assert excerpt_supports_claim(claim, d1_3, question=question)
    assert not excerpt_supports_claim(claim, d1_7, question=question)


def test_timing_claim_does_not_keep_same_topic_citation():
    excerpts = (
        PromptExcerpt(
            "D1:3",
            "user",
            "Priya: I went to a chess practice group yesterday and it was so useful.",
            ("mem:D1:3:c0",),
        ),
        PromptExcerpt(
            "D1:7",
            "user",
            "Priya: The practice group has taught me patience and given me confidence to enter a tournament.",
            ("mem:D1:7:c0",),
        ),
    )
    question = "When did Priya go to the chess practice group?"
    raw = (
        "Priya went to the chess practice group yesterday according to her "
        "statement in message D1 [D1:7]."
    )
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert diag["rawAnswer"] == raw
    assert "[D1:7]" not in display
    assert "[D1:3]" not in display
    assert diag["citations"][0]["id"] == "D1:7"
    assert diag["citations"][0]["kept"] is False
    assert diag["citations"][0]["reason"] == "unsupported_excerpt"
    assert display == "Priya went to the chess practice group yesterday."
    assert "(unsupported)" not in display
    assert "according to" not in display.lower()
    assert "statement" not in display.lower()


def test_timing_claim_keeps_excerpt_that_states_the_time():
    excerpts = (
        PromptExcerpt(
            "D1:3",
            "user",
            "Priya: I went to a chess practice group yesterday and it was so useful.",
            ("mem:D1:3:c0",),
        ),
        PromptExcerpt(
            "D1:7",
            "user",
            "Priya: The practice group has taught me patience and given me confidence to enter a tournament.",
            ("mem:D1:7:c0",),
        ),
    )
    raw = "Priya went to the chess practice group yesterday [D1:3]."
    display, diag = validate_answer_citations(
        raw,
        excerpts,
        question="When did Priya go to the chess practice group?",
    )
    assert display == raw
    assert diag["citations"][0]["kept"] is True
    assert diag["citations"][0]["id"] == "D1:3"


def test_excerpt_support_rejects_unrelated_retrieved_id():
    question = "What fields would Priya be likely to pursue in her education?"
    oceanography = "Priya is interested in oceanography or marine science"
    assert excerpt_supports_claim(
        oceanography,
        "Priya: I'm keen on oceanography or working in marine science.",
        question=question,
    )
    assert not excerpt_supports_claim(
        oceanography,
        "Priya: Cool! What did it look like?",
        question=question,
    )
    assert excerpt_supports_claim(
        "She wants to continue her education and explore career options",
        "Priya: Going to continue my edu and check out career options.",
        question=question,
    )


def test_model_cannot_attribute_marine_health_to_unrelated_prompt_id():
    ctx = _education_context()
    assert {"D1:9", "D1:11", "D10:15"} <= {item.message_id for item in ctx.excerpts}

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            assert "[D1:11]" in prompt
            assert "[D10:15]" in prompt
            return (
                "Priya is likely to pursue oceanography or marine science [D10:15]. "
                "She also wants to continue her education [D1:9]."
            )

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    raw = meta["rawAnswer"]
    assert "[D10:15]" in raw
    assert "[D10:15]" not in text
    assert "(unsupported)" not in text
    assert "[D1:9]" in text
    assert "oceanography" in text.lower()
    assert any(row["id"] == "D10:15" and row["reason"] == "unsupported_excerpt" for row in meta["citations"])
    assert any(row["id"] == "D1:9" and row["kept"] for row in meta["citations"])


def test_prompt_binds_each_message_id_to_its_excerpt_text():
    ctx = _education_context()
    assert "(role:" not in ctx.prompt
    for item in ctx.excerpts:
        assert f"[{item.message_id}] {item.content}" in ctx.prompt
        for memory_id in item.memory_ids:
            assert f"[{memory_id}]" not in ctx.prompt
            assert memory_id in ctx.prompt


def test_relevant_citations_remain_intact():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return (
                "Priya wants to continue her education and explore career options [D1:9]. "
                "She is interested in oceanography or marine science [D1:11]."
            )

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert "[D1:9]" in text
    assert "[D1:11]" in text
    assert "[D10:15]" not in text
    assert "unsupported" not in text.lower()
    assert meta["rawAnswer"] == text or "[D1:11]" in meta["rawAnswer"]
    assert not meta["strippedCitations"]
    assert meta["rawAnswer"]


def test_oceanography_claim_is_cited_with_supporting_d1_11():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            assert "[D1:11] Priya: I'm keen on oceanography" in prompt
            assert "[D10:15] Priya: Cool! What did it look like?" in prompt
            return "Priya is interested in oceanography or working in marine science [D1:11]."

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert "[D1:11]" in text
    assert "[D10:15]" not in text
    assert "unsupported" not in text.lower()
    assert text == meta["rawAnswer"]


def test_education_claim_is_cited_with_supporting_d1_9():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return "Priya wants to continue her education and explore career options [D1:9]."

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert "[D1:9]" in text
    assert "[D10:15]" not in text
    assert "unsupported" not in text.lower()


def test_multiple_claims_keep_distinct_supporting_sources():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return (
                "She will continue her education and look at career options [D1:9]. "
                "She is keen on oceanography or marine science [D1:11]."
            )

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert text.index("[D1:9]") < text.index("[D1:11]")
    assert "[D10:15]" not in text
    assert not meta["strippedCitations"]


def test_missing_citation_is_not_invented():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return "Priya is interested in oceanography or marine science."

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert text == "Priya is interested in oceanography or marine science."
    assert "[D1:11]" not in text
    assert "[D1:9]" not in text
    assert "[D10:15]" not in text
    assert meta["rawAnswer"] == text
    assert not meta.get("strippedCitations")


def _priya_excerpts():
    return (
        PromptExcerpt(
            "D1:9",
            "user",
            "Priya: Going to continue my edu and check out career options, which is pretty exciting!",
            ("mem:D1:9:c0",),
        ),
        PromptExcerpt(
            "D1:11",
            "user",
            "Priya: I'm keen on oceanography or working in marine science - I'd love to protect the reefs near my home.",
            ("mem:D1:11:c0",),
        ),
        PromptExcerpt("D10:15", "user", "Priya: Cool! What did it look like?", ("mem:D10:15:c0",)),
    )


def test_qwen7b_trailing_cite_after_period_keeps_supporting_d1_11():
    excerpts = _priya_excerpts()
    question = "What fields would Priya be likely to pursue in her education?"
    raw = "Priya would likely pursue oceanography or working in marine science. [D1:11]"
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert diag["rawAnswer"] == raw
    assert display == raw
    assert "[D1:11]" in display
    assert diag["citations"][0]["id"] == "D1:11"
    assert diag["citations"][0]["kept"] is True
    assert "oceanography" in diag["citations"][0]["claim"]
    assert not diag["stripped"]


def test_citation_after_question_or_exclamation_uses_preceding_sentence():
    excerpts = _priya_excerpts()
    question = "What fields would Priya be likely to pursue in her education?"
    for raw in (
        "Would she pursue oceanography or marine science? [D1:11]",
        "She is keen on oceanography or marine science! [D1:11]",
    ):
        display, diag = validate_answer_citations(raw, excerpts, question=question)
        assert display == raw
        assert diag["citations"][0]["kept"] is True
        assert diag["citations"][0]["id"] == "D1:11"


def test_trailing_cite_attaches_to_last_sentence_only():
    excerpts = _priya_excerpts()
    question = "What fields would Priya be likely to pursue in her education?"
    raw = (
        "She wants to continue her education and explore career options. "
        "She is interested in oceanography or marine science. [D1:11]"
    )
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert display == raw
    assert diag["citations"][0]["kept"] is True
    assert "oceanography" in diag["citations"][0]["claim"]
    assert "career options" not in diag["citations"][0]["claim"]


def test_trailing_cite_still_rejects_unsupported_excerpt():
    excerpts = _priya_excerpts()
    question = "What fields would Priya be likely to pursue in her education?"
    raw = "Priya would likely pursue oceanography or working in marine science. [D10:15]"
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert diag["rawAnswer"] == raw
    assert "[D10:15]" not in display
    assert "[D1:11]" not in display
    assert diag["citations"][0]["id"] == "D10:15"
    assert diag["citations"][0]["kept"] is False
    assert diag["citations"][0]["reason"] == "unsupported_excerpt"
    assert "oceanography" in display.lower()


def test_trailing_cite_outside_prompt_is_dropped():
    excerpts = _priya_excerpts()
    raw = "Priya would likely pursue oceanography or working in marine science. [D99:1]"
    display, diag = validate_answer_citations(
        raw,
        excerpts,
        question="What fields would Priya be likely to pursue in her education?",
    )
    assert "[D99:1]" not in display
    assert "[D1:11]" not in display
    assert diag["stripped"][0]["reason"] == "not_in_prompt"


def test_valid_excerpt_presentation_is_kept():
    excerpts = (
        PromptExcerpt(
            "D1:11", "user", "Priya: I'm keen on oceanography or working in marine science.", ("m1",),
        ),
        PromptExcerpt("D10:15", "user", "Priya: Cool! What did it look like?", ("m2",)),
    )
    question = "What fields would Priya be likely to pursue in her education?"
    raw = (
        "She is interested in oceanography or marine science [D1:11]. "
        "The relevant excerpt [D1:11] states "
        "\"Priya: I'm keen on oceanography or working in marine science.\""
    )
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert "[D1:11]" in display
    assert "keen on oceanography" in display
    assert "D10:15" not in display
    assert "unsupported" not in display.lower()
    assert diag["rawAnswer"] == raw


def test_invalid_citation_is_not_rewritten_to_another_source():
    excerpts = (
        PromptExcerpt(
            "D1:11", "user", "Priya: I'm keen on oceanography or working in marine science.", ("m1",),
        ),
        PromptExcerpt("D10:15", "user", "Priya: Cool! What did it look like?", ("m2",)),
    )
    question = "What fields would Priya be likely to pursue in her education?"
    raw = "Priya is interested in oceanography [D10:15]."
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert "[D1:11]" not in display
    assert "[D10:15]" not in display
    assert diag["rawAnswer"] == raw
    assert display != raw
    assert "(unsupported)" not in display
    assert "oceanography" in display.lower()


def test_citation_outside_prompt_is_dropped():
    excerpts = (
        PromptExcerpt(
            "D1:9", "user", "Priya: Going to continue my edu and check out career options.", ("m1",),
        ),
    )
    display, diag = validate_answer_citations(
        "She mentioned career options [D99:1].",
        excerpts,
        question="What fields would Priya be likely to pursue in her education?",
    )
    assert "[D99:1]" not in display
    assert "[D1:9]" not in display
    assert diag["stripped"][0]["reason"] == "not_in_prompt"


def test_rejected_excerpt_embedded_in_prose_is_not_shown_as_evidence():
    excerpts = (
        PromptExcerpt(
            "D1:11", "user", "Priya: I'm keen on oceanography or working in marine science.", ("m1",),
        ),
        PromptExcerpt("D10:15", "user", "Priya: Cool! What did it look like?", ("m2",)),
    )
    question = "What fields would Priya be likely to pursue in her education?"
    raw = (
        "Priya is likely to pursue oceanography or marine science [D10:15]. "
        "The relevant excerpt [D10:15] states \"Priya: Cool! What did it look like?\"."
    )
    display, diag = validate_answer_citations(raw, excerpts, question=question)
    assert diag["rawAnswer"] == raw
    assert "[D10:15]" not in display
    assert "D10:15" not in display
    assert "Cool! What did it look like?" not in display
    assert "The relevant excerpt" not in display
    assert "[D1:11]" not in display
    assert "(unsupported)" not in display
    assert "oceanography" in display.lower()


def test_model_prose_quoting_rejected_id_does_not_replace_it_with_valid_source():
    ctx = _education_context()

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return (
                "Priya is interested in oceanography or marine science [D10:15]. "
                "The relevant excerpt D10:15 states \"Priya: Cool! What did it look like?\". "
                "She also wants to continue her education [D1:9]."
            )

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert meta["rawAnswer"]
    assert "D10:15" in meta["rawAnswer"]
    assert "Cool! What did it look like?" in meta["rawAnswer"]
    assert "D10:15" not in text
    assert "Cool! What did it look like?" not in text
    assert "The relevant excerpt" not in text
    assert "[D1:11]" not in text
    assert "[D1:9]" in text
    assert "(unsupported)" not in text


LIVE_COPIED_PROMPT_RAW = (
    "Based on the source excerpts provided, Priya is interested in careers that "
    "involve helping others with marine science issues. The relevant excerpt from "
    "[D10:15] states:\n"
    "\n"
    "[D10:15] (role: user; memories: mem:D10:15:c0)\n"
    "Priya: Cool! What did it look like?\n"
    "\n"
    "This indicates Priya is considering oceanography or working in marine science "
    "as a field of interest. The source also mentions that Priya has expressed "
    "interest in \"protecting the reefs near her home,\" further supporting this "
    "pursuit."
)

LIVE_EXCERPTS = (
    PromptExcerpt(
        "D1:9",
        "user",
        "Priya: Going to continue my edu and check out career options, which is pretty exciting!",
        ("mem:D1:9:c0",),
    ),
    PromptExcerpt(
        "D1:11",
        "user",
        "Priya: I'm keen on oceanography or working in marine science - I'd love to protect the reefs near my home.",
        ("mem:D1:11:c0",),
    ),
    PromptExcerpt(
        "D10:15",
        "user",
        "Priya: Cool! What did it look like?",
        ("mem:D10:15:c0",),
    ),
)


def test_copied_prompt_block_after_unsupported_marker_is_not_displayed():
    question = "What fields would Priya be likely to pursue in her education?"
    display, diag = validate_answer_citations(
        LIVE_COPIED_PROMPT_RAW, LIVE_EXCERPTS, question=question,
    )
    assert diag["rawAnswer"] == LIVE_COPIED_PROMPT_RAW
    assert "D10:15" not in display
    assert "mem:D10:15:c0" not in display
    assert "Cool! What did it look like?" not in display
    assert "(role: user; memories:" not in display
    assert "The relevant excerpt" not in display
    assert "oceanography" in display.lower()
    assert "reefs near her home" in display
    assert "[D1:11]" not in display


def test_copied_prompt_block_preserves_valid_d1_11_and_is_used_as_answer_content():
    question = "What fields would Priya be likely to pursue in her education?"
    raw = LIVE_COPIED_PROMPT_RAW.replace(
        "marine science issues.",
        "marine science issues [D1:11].",
    )
    ctx = _education_context()
    ctx.excerpts = LIVE_EXCERPTS

    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt, **kwargs):
            return raw

    text, meta = generate_answer(
        ctx, GenerationSettings(enabled=True, model="fake-reader"), Fake(),
    )
    assert meta["rawAnswer"] == raw
    assert "D10:15" in meta["rawAnswer"]
    assert "mem:D10:15:c0" in meta["rawAnswer"]
    assert "Cool! What did it look like?" in meta["rawAnswer"]
    assert "D10:15" not in text
    assert "mem:D10:15:c0" not in text
    assert "Cool! What did it look like?" not in text
    assert "[D1:11]" in text
    assert "oceanography" in text.lower()
    assert text != meta["rawAnswer"]


def test_missing_model_explains_configuration(monkeypatch):
    monkeypatch.delenv("CONTINUATOR_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("CONTINUATOR_ANSWER_MODEL", raising=False)
    monkeypatch.setenv("CONTINUATOR_ANSWER_ENABLED", "1")
    settings = load_generation_settings()
    assert settings.configured is False
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace._generation = settings
    result = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=True)
    assert result["answer"]["content"] == NOT_CONFIGURED_ANSWER
    assert result["answer"]["evidence"]["generation"]["status"] == "not_configured"
    assert result["answer"]["evidence"]["memoryIds"]


# ---- message dates and derived relative-date notes ------------------------------------------------


def _dated_conversation(*messages: tuple[str, str, str | None]) -> Conversation:
    return Conversation("dated-1", tuple(
        Message(mid, "dated-1", i, "user", text, ts) for i, (mid, text, ts) in enumerate(messages)))


def _dated_context(conversation: Conversation, question: str = "When did Sam go to the workshop?"):
    ids = [m.message_id for m in conversation.messages]
    evidence = _evidence(selected=ids, matches=[_match(f"mem:{mid}", "", [mid], rank=i + 1)
                                                for i, mid in enumerate(ids)])
    return build_answer_context(question, evidence, conversation, seed_ids=[], expanded_ids=[],
                                max_context_chars=12_000)


def test_excerpts_carry_their_own_message_date_and_note():
    conv = _dated_conversation(
        ("s1", "Sam: I went to a pottery workshop yesterday.", "2024-03-12T10:00:00"),
        ("s2", "Riley: I went to a pottery workshop yesterday too!", "2024-08-01T09:30:00"),
    )
    ctx = _dated_context(conv)
    a, b = ctx.excerpts
    assert (a.message_id, a.content, a.timestamp) == ("s1", conv.messages[0].content, "2024-03-12T10:00:00")
    assert (b.message_id, b.content, b.timestamp) == ("s2", conv.messages[1].content, "2024-08-01T09:30:00")
    assert [r.value for r in a.date_resolutions] == ["11 March 2024"]
    assert [r.value for r in b.date_resolutions] == ["31 July 2024"]
    assert ("[s1] (message date: 12 March 2024) Sam: I went to a pottery workshop yesterday.\n"
            '(derived from the message date, not part of the message: "yesterday" = 11 March 2024)\n'
            "linked memory mem:s1") in ctx.prompt
    assert ("[s2] (message date: 1 August 2024) Riley: I went to a pottery workshop yesterday too!\n"
            '(derived from the message date, not part of the message: "yesterday" = 31 July 2024)\n'
            "linked memory mem:s2") in ctx.prompt


def test_original_text_ids_and_citation_markers_are_unchanged():
    conv = _dated_conversation(("s1", "Sam: Last Friday I saw [the] show.", "2024-03-12T10:00:00"))
    ctx = _dated_context(conv)
    assert ctx.excerpts[0].content == conv.messages[0].content
    note = ctx.excerpts[0].date_note
    assert note and "[" not in note and "]" not in note
    headers = [line.split("]")[0] + "]" for line in ctx.prompt.splitlines() if line.startswith("[s")]
    assert headers == ["[s1]"]


@pytest.mark.parametrize("timestamp", [None, "", "1:56 pm on 12 March, 2024", "soon"])
def test_missing_or_invalid_timestamp_adds_no_date_and_no_note(timestamp):
    ctx = _dated_context(_dated_conversation(("s1", "Sam: I went yesterday.", timestamp)))
    assert ctx.excerpts[0].message_date is None and ctx.excerpts[0].date_resolutions == ()
    assert "[s1] Sam: I went yesterday." in ctx.prompt
    assert "message date" not in ctx.prompt and "derived from the message date" not in ctx.prompt


def test_idioms_get_no_note_and_qualifiers_and_plans_are_labelled():
    conv = _dated_conversation(
        ("s1", "Sam: Wow, it feels like just yesterday I bought it!", "2024-03-12T10:00:00"),
        ("s2", "Riley: I started about five years ago.", "2024-03-12T10:01:00"),
        ("s3", "Sam: We're going camping next month.", "2024-03-12T10:02:00"),
    )
    s1, s2, s3 = _dated_context(conv).excerpts
    assert s1.date_note is None
    assert s2.date_note == '(derived from the message date, not part of the message: "about five years ago" = around 2019)'
    assert s3.date_note == ('(derived from the message date, not part of the message: '
                            '"next month" = April 2024 (still ahead at the message date))')


def test_image_captions_are_not_resolved_but_message_text_is():
    conv = _dated_conversation(("s1", "Sam: Look at this!\n[image: a photo taken yesterday]", "2024-03-12T10:00:00"))
    assert _dated_context(conv).excerpts[0].date_resolutions == ()


def _excerpts_for_citations():
    conv = _dated_conversation(
        ("s1", "Sam: I went to a pottery workshop yesterday.", "2024-03-12T10:00:00"),
        ("s2", "Riley: We hiked last year.", "2024-08-01T09:30:00"),
    )
    return _dated_context(conv).excerpts


def test_date_claim_supported_by_its_own_note_keeps_its_citation():
    excerpts = _excerpts_for_citations()
    q = "When did Sam go to the workshop?"
    for answer in ("On 11 March 2024 [s1].", "Sam went to the workshop on 11 March 2024 [s1].",
                   "It was March 11, 2024 [s1].", "Riley hiked in 2023 [s2]."):
        cleaned, meta = validate_answer_citations(answer, excerpts, question=q)
        assert [c["kept"] for c in meta["citations"]] == [True], (answer, meta["citations"])
        assert "[s" in cleaned


def test_same_claim_without_the_derived_note_is_still_rejected():
    """Control: the support comes from the note, not from loosened text matching."""
    plain = tuple(PromptExcerpt(e.message_id, e.role, e.content, e.memory_ids) for e in _excerpts_for_citations())
    _, meta = validate_answer_citations("On 11 March 2024 [s1].", plain, question="When did Sam go to the workshop?")
    assert meta["citations"][0]["kept"] is False


def test_derived_note_does_not_leak_to_other_excerpts_or_unrelated_claims():
    excerpts = _excerpts_for_citations()
    q = "When did Sam go to the workshop?"
    for answer in (
        "On 11 March 2024 [s2].",                     # the date belongs to s1's note, not s2's
        "On 12 March 2024 [s1].",                     # wrong day for s1's note
        "Sam adopted a puppy on 11 March 2024 [s1].",  # right date, facts not in s1
        "On 3 June 2023 [s2].",                       # more specific than s2's "2023"
    ):
        _, meta = validate_answer_citations(answer, excerpts, question=q)
        assert [c["kept"] for c in meta["citations"]] == [False], (answer, meta["citations"])


def _kept(answer: str, excerpts, question: str) -> list[bool]:
    _, meta = validate_answer_citations(answer, excerpts, question=question)
    return [c["kept"] for c in meta["citations"]]


def _approx_and_plan_excerpts():
    conv = _dated_conversation(
        ("s1", "Riley: I started painting about five years ago.", "2024-03-12T10:00:00"),
        ("s2", "Sam: We're going camping next month.", "2024-03-12T10:01:00"),
        ("s3", "Sam: We went camping last month.", "2024-05-20T10:00:00"),
    )
    return _dated_context(conv).excerpts


@pytest.mark.parametrize("answer", [
    "Riley started painting around 2019 [s1].",
    "Riley started painting in about 2019 [s1].",
    "Riley started painting approximately in 2019 [s1].",
    "Riley started painting in 2019 or so [s1].",
])
def test_approximate_note_supports_a_hedged_date_claim(answer):
    assert _kept(answer, _approx_and_plan_excerpts(), "When did Riley start painting?") == [True]


@pytest.mark.parametrize("question", ["When did Riley start painting?", "What does Riley do?"])
@pytest.mark.parametrize("answer", [
    "Riley started painting in 2019 [s1].",
    "In 2019 [s1].",
    "About painting: Riley started in 2019 [s1].",  # "about" is not a hedge on this date
])
def test_approximate_note_does_not_support_a_definite_date_claim(answer, question):
    # "What does Riley do?" leaves "painting" as overlapping text; it must not vouch for the date.
    assert _kept(answer, _approx_and_plan_excerpts(), question) == [False]


@pytest.mark.parametrize("answer", [
    "Sam plans to go camping in April 2024 [s2].",
    "Sam is going camping in April 2024 [s2].",
    "Sam will go camping in April 2024 [s2].",
])
def test_future_note_supports_a_plan_claim(answer):
    assert _kept(answer, _approx_and_plan_excerpts(), "When is Sam going camping?") == [True]


@pytest.mark.parametrize("question", ["When did Sam go camping?", "What are Sam's plans?"])
@pytest.mark.parametrize("answer", [
    "Sam went camping in April 2024 [s2].",
    "Sam camped in April 2024 [s2].",
    "In April 2024 [s2].",
])
def test_future_note_does_not_support_a_completed_event_claim(answer, question):
    assert _kept(answer, _approx_and_plan_excerpts(), question) == [False]


def test_past_note_still_supports_a_completed_event_claim():
    excerpts = _approx_and_plan_excerpts()
    assert excerpts[2].date_resolutions[0].relation == "past"
    assert _kept("Sam went camping in April 2024 [s3].", excerpts, "When did Sam go camping?") == [True]
    assert _kept("Sam went camping in April 2024 [s2] [s3].", excerpts, "When did Sam go camping?") == [False, True]


def test_date_written_in_the_text_is_not_affected_by_the_note_rules():
    conv = _dated_conversation(("s1", "Sam: In 2019 I moved here; about five years ago, roughly.",
                                "2024-03-12T10:00:00"))
    excerpts = _dated_context(conv).excerpts
    assert excerpts[0].date_resolutions
    assert _kept("Sam moved in 2019 [s1].", excerpts, "When did Sam move?") == [True]


def test_excerpt_support_without_resolutions_is_unchanged():
    excerpt = "Sam: I went to a pottery workshop yesterday."
    assert excerpt_supports_claim("Sam went to the pottery workshop", excerpt) is True
    assert excerpt_supports_claim("On 11 March 2024", excerpt) is False


# ---- answer evidence count ---------------------------------------------------------------------


def test_answer_top_k_default_env_and_explicit():
    from continuator_memory.webapi.answering import (
        ANSWER_TOP_K_ENV, DEFAULT_ANSWER_TOP_K, MAX_ANSWER_TOP_K, load_answer_top_k)
    assert DEFAULT_ANSWER_TOP_K == 25
    assert load_answer_top_k(env={}) == 25
    assert load_answer_top_k(env={ANSWER_TOP_K_ENV: " 12 "}) == 12
    assert load_answer_top_k(5, env={ANSWER_TOP_K_ENV: "30"}) == 5  # explicit smaller limit wins
    assert load_answer_top_k(15, env={}) == 15
    assert load_answer_top_k(MAX_ANSWER_TOP_K, env={}) == MAX_ANSWER_TOP_K
    for bad_env in ("0", "33", "ten", "2.5"):
        with pytest.raises(ValueError):
            load_answer_top_k(env={ANSWER_TOP_K_ENV: bad_env})
    for bad in (0, -1, MAX_ANSWER_TOP_K + 1):
        with pytest.raises(ValueError):
            load_answer_top_k(bad, env={})


def test_workspace_ceiling_matches_answer_ceiling():
    from continuator_memory.webapi.answering import MAX_ANSWER_TOP_K
    from continuator_memory.webapi.workspace import MAX_K
    assert MAX_K == MAX_ANSWER_TOP_K
