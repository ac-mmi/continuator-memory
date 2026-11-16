#!/usr/bin/env python3
"""Build the JSON body for POST /import from examples/quickstart/conversation.txt."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
text = (ROOT / "conversation.txt").read_text(encoding="utf-8")
print(
    json.dumps(
        {
            "text": text,
            "filename": "conversation.txt",
            "extract": "messages",
        }
    )
)
