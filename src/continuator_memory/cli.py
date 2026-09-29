"""Command-line interface: ingest, retrieve, ask, evaluate, memories."""

from __future__ import annotations

import argparse
import json
import sys

from continuator_memory.config import Settings
from continuator_memory.generation import GenerationError, OllamaGenerator
from continuator_memory.models import ParseError, RetrievedChunk, load_conversation
from continuator_memory.pipeline import Pipeline
from continuator_memory.store import Store

MEMORY_HELP = "Phase 3: use memory formation / memory-aware retrieval"


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="continuator-memory")
    parser.add_argument(
        "--db", default=str(settings.db_path), help=f"SQLite path (default: {settings.db_path})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a conversation JSON file")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--memory", action="store_true", help=MEMORY_HELP)

    for name, help_text in (
        ("retrieve", "show the top-k chunks retrieved for a question"),
        ("ask", "answer a question using retrieved chunks and Ollama"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question")
        p.add_argument("-c", "--conversation", help="restrict to one conversation ID")
        p.add_argument("-k", "--top-k", type=int, default=settings.top_k)
        p.add_argument("--json", action="store_true", help="emit JSON")
        p.add_argument("--memory", action="store_true", help=MEMORY_HELP)
        if name == "ask":
            p.add_argument(
                "--show-chunks", action="store_true", help="also print retrieved chunks"
            )

    p_eval = sub.add_parser(
        "evaluate", help="run a benchmark dataset against a fresh in-memory index"
    )
    p_eval.add_argument("dataset", help="benchmark dataset JSON")
    p_eval.add_argument(
        "-o", "--output-dir", default="benchmarks/results", help="where to write JSON/Markdown"
    )
    p_eval.add_argument(
        "-k", "--top-k", type=int, default=settings.top_k,
        help=f"chunks given to the LLM (default: {settings.top_k})",
    )
    p_eval.add_argument(
        "--retrieval-only", action="store_true", help="skip answer generation (no Ollama calls)"
    )
    p_eval.add_argument("--limit", type=int, help="evaluate only the first N questions")
    p_eval.add_argument("--memory", action="store_true", help=MEMORY_HELP)

    p_mem = sub.add_parser("memories", help="show stored memories and relationships (Phase 3)")
    p_mem.add_argument("-c", "--conversation", help="restrict to one conversation ID")
    p_mem.add_argument("--current-only", action="store_true", help="hide superseded records")
    p_mem.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def _build_pipeline(store: Store, settings: Settings, memory: bool):
    from continuator_memory.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    generator = OllamaGenerator(settings.ollama_model, settings.ollama_host)
    if not memory:
        return Pipeline(store, embedder, generator, settings.window_size, settings.overlap)
    from continuator_memory.memory.extraction import MemoryExtractor, OllamaJSONClient
    from continuator_memory.memory.pipeline import MemoryPipeline

    extractor = MemoryExtractor(OllamaJSONClient(settings.extraction_model, settings.ollama_host))
    return MemoryPipeline(
        store, embedder, generator, extractor, settings.window_size, settings.overlap
    )


def _evaluate(args: argparse.Namespace, settings: Settings) -> int:
    from continuator_memory.evaluation.dataset import DatasetError, load_dataset
    from continuator_memory.evaluation.harness import run_evaluation
    from continuator_memory.evaluation.report import write_results

    try:
        dataset = load_dataset(args.dataset)
    except (OSError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with Store(":memory:") as store:
        pipeline = _build_pipeline(store, settings, args.memory)
        config = {
            "pipeline": "phase3-memory" if args.memory else "phase1-baseline",
            "embedding_model": settings.embedding_model,
            "ollama_model": None if args.retrieval_only else settings.ollama_model,
            "ollama_host": settings.ollama_host,
            "window_size": settings.window_size,
            "overlap": settings.overlap,
        }
        if args.memory:
            config["extraction_model"] = settings.extraction_model
        results = run_evaluation(
            dataset,
            pipeline,
            generation_top_k=args.top_k,
            retrieval_only=args.retrieval_only,
            limit=args.limit,
            config=config,
            progress=lambda msg: print(msg, file=sys.stderr),
        )

    json_path, md_path = write_results(results, args.output_dir)
    retrieval = results["aggregate"]["retrieval"]["overall"]
    answer = results["aggregate"]["answer"]["overall"]
    print(
        "retrieval: n={n} MRR={mrr} ".format(n=retrieval["n"], mrr=_f(retrieval["mrr"]))
        + " ".join(f"R@{k}={_f(v)}" for k, v in retrieval["recall_at"].items())
    )
    if not args.retrieval_only:
        print(f"answers: accuracy={_f(answer['accuracy'])} ({answer['n_correct']}/{answer['n']})")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def _memories(args: argparse.Namespace) -> int:
    from continuator_memory.memory.store import MemoryStore

    with Store(args.db) as store:
        ms = MemoryStore(store.connection)
        memories = ms.memories(args.conversation, current_only=args.current_only)
        relationships = ms.relationships(args.conversation, current_only=args.current_only)
        if args.json:
            print(json.dumps(
                {"memories": [m.__dict__ for m in memories],
                 "relationships": [r.__dict__ for r in relationships]},
                indent=2, default=list,
            ))
            return 0
        if not memories and not relationships:
            print("(no memories; ingest with --memory first)")
        current_subject = None
        for m in sorted(memories, key=lambda m: (m.conversation_id, m.entity_id, m.predicate,
                                                 not m.is_current, -m.observed_position)):
            if (m.conversation_id, m.subject) != current_subject:
                current_subject = (m.conversation_id, m.subject)
                print(f"\n{m.subject}  [{m.conversation_id}]")
            status = "CURRENT" if m.is_current else "SUPERSEDED"
            extra = f" supersedes={m.supersedes_memory_id}" if m.supersedes_memory_id else ""
            print(f"  {m.memory_id}  {m.predicate} = {m.value}  [{status}]  "
                  f"messages={','.join(m.source_message_ids)}{extra}")
        if relationships:
            print("\nRelationships")
        for r in relationships:
            status = "CURRENT" if r.is_current else "SUPERSEDED"
            print(f"  {r.relationship_id}  {r.subject} -> {r.predicate} -> {r.object}  "
                  f"[{status}]  messages={','.join(r.source_message_ids)}")
    return 0


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_chunks(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        print("(no chunks retrieved)")
    for c in chunks:
        print(f"#{c.rank}  score={c.score:.4f}  chunk={c.chunk_id}  conversation={c.conversation_id}")
        print(f"    messages: {', '.join(c.source_message_ids)}")
        for line in c.text.splitlines():
            print(f"    | {line}")
        print()


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    args = _build_parser(settings).parse_args(argv)

    if args.command == "evaluate":
        return _evaluate(args, settings)
    if args.command == "memories":
        return _memories(args)

    with Store(args.db) as store:
        pipeline = _build_pipeline(store, settings, args.memory)

        if args.command == "ingest":
            try:
                conversation = load_conversation(args.path)
            except (OSError, ParseError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            chunks = pipeline.ingest(conversation)
            print(
                f"ingested conversation '{conversation.conversation_id}': "
                f"{len(conversation.messages)} messages, {len(chunks)} chunks "
                f"(embedding model: {settings.embedding_model}, db: {args.db})"
            )
            if args.memory:
                print(f"memory formation: {json.dumps(pipeline.reports[-1].to_dict())}")
            return 0

        if args.command == "retrieve":
            if args.memory:
                from continuator_memory.memory.context import render_memory_context

                retrieval = pipeline.retrieve_memory(args.question, args.top_k, args.conversation)
                if args.json:
                    print(json.dumps({"memory": retrieval.trace(),
                                      "chunks": [c.to_dict() for c in retrieval.chunks]},
                                     indent=2))
                else:
                    print(render_memory_context(retrieval) + "\n")
                    _print_chunks(retrieval.chunks)
                return 0
            chunks = pipeline.retrieve(args.question, args.top_k, args.conversation)
            if args.json:
                print(json.dumps([c.to_dict() for c in chunks], indent=2))
            else:
                _print_chunks(chunks)
            return 0

        if args.command == "ask":
            try:
                answer = pipeline.ask(args.question, args.top_k, args.conversation)
            except GenerationError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if args.json:
                payload = {
                    "question": answer.question,
                    "answer": answer.answer,
                    "model": answer.model_name,
                    "retrieved": [c.to_dict() for c in answer.retrieved],
                }
                if args.memory:
                    payload["memory"] = answer.memory_trace
                print(json.dumps(payload, indent=2))
            else:
                if args.show_chunks:
                    if args.memory:
                        print(answer.prompt + "\n")
                    else:
                        _print_chunks(answer.retrieved)
                print(answer.answer)
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
