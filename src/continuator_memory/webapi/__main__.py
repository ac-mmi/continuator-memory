from __future__ import annotations

import argparse
import os

from continuator_memory.webapi.server import DEFAULT_HOST, DEFAULT_PORT, serve_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m continuator_memory.webapi")
    parser.add_argument("--host", default=os.environ.get("CONTINUATOR_WEB_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTINUATOR_WEB_PORT", DEFAULT_PORT)))
    parser.add_argument("--sample", choices=("fixture", "db"), default=os.environ.get("CONTINUATOR_WEB_SAMPLE", "fixture"))
    parser.add_argument("--db", default=os.environ.get("CONTINUATOR_DB_PATH"))
    parser.add_argument("--classifications", default=os.environ.get("CONTINUATOR_BPRIME_CLASSIFICATIONS"))
    parser.add_argument("--conversation-id", default=os.environ.get("CONTINUATOR_CONVERSATION_ID"))
    parser.add_argument("-k", "--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    serve_from_env(
        host=args.host,
        port=args.port,
        sample=args.sample,
        db=args.db,
        classifications=args.classifications,
        conversation_id=args.conversation_id,
        k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
