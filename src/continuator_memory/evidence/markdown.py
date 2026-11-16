"""Structure-first Markdown → section segments with heading ancestry and char offsets.

No third-party Markdown dependency. Heading detection matches ATX-style ``#`` headings
(same family as ``knowledge/documents.py``).

Oversized sections: call ``split_section_text`` (optional max_chars). The default is no
split; callers can plug a different strategy later without changing the Document model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    """One structural section: heading path + body text + source span."""

    heading_path: tuple[str, ...]
    text: str
    char_start: int
    char_end: int
    heading_level: int  # 0 = preamble before any heading


def parse_markdown_sections(source: str) -> list[MarkdownSection]:
    """Split Markdown into sections at heading boundaries, preserving ancestry.

    Heading stack rules:
    - A heading at level N closes any open headings at level >= N.
    - Body lines accumulate under the current stack until the next heading.
    - Preamble (text before the first heading) is emitted as ``heading_path=()``.
    - Heading-only sections with empty body are skipped.
    """
    lines = source.splitlines(keepends=True)
    if not lines and source:
        # No newlines; treat whole string as one preamble body.
        text = source.strip()
        return [MarkdownSection((), text, 0, len(source), 0)] if text else []

    stack: list[tuple[int, str]] = []  # (level, title)
    sections: list[MarkdownSection] = []
    body_parts: list[str] = []
    body_start: int | None = None
    offset = 0
    current_level = 0

    def flush(end: int) -> None:
        nonlocal body_parts, body_start
        if body_start is None:
            body_parts = []
            return
        raw = "".join(body_parts)
        text = raw.strip()
        if text:
            # Trim leading/trailing whitespace in the reported span when possible.
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            start = body_start + lead
            stop = end - trail
            if stop < start:
                start, stop = body_start, end
            path = tuple(title for _, title in stack)
            sections.append(
                MarkdownSection(
                    heading_path=path,
                    text=text,
                    char_start=start,
                    char_end=stop,
                    heading_level=current_level,
                )
            )
        body_parts = []
        body_start = None

    for line in lines:
        line_start = offset
        offset += len(line)
        stripped = line.rstrip("\n\r")
        match = _HEADING.match(stripped)
        if match:
            flush(line_start)
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_level = level
            continue
        if body_start is None:
            body_start = line_start
        body_parts.append(line)

    flush(offset)
    return sections


def build_retrieval_text(heading_path: tuple[str, ...], text: str) -> str:
    """Deterministic retrieval representation: optional ancestry header + body."""
    body = (text or "").strip()
    if heading_path:
        ancestry = " > ".join(heading_path)
        return f"{ancestry}\n\n{body}" if body else ancestry
    return body


def section_id_for(heading_path: tuple[str, ...]) -> str:
    """Stable section key for section-level evaluation gold."""
    if not heading_path:
        return "sec:preamble"
    parts = [_slug(part) for part in heading_path]
    return "sec:" + "/".join(parts)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "section"


def split_section_text(text: str, *, max_chars: int | None = None) -> list[str]:
    """Optional oversized-section splitter (extension point).

    Default: return the section as a single piece.
    When ``max_chars`` is set, split on paragraph boundaries preferentially,
    then hard-split remaining runs. No overlap.
    """
    body = (text or "").strip()
    if not body:
        return []
    if max_chars is None or max_chars <= 0 or len(body) <= max_chars:
        return [body]

    paragraphs = re.split(r"\n\s*\n", body)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        if current:
            pieces.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            emit()
            for i in range(0, len(para), max_chars):
                pieces.append(para[i : i + max_chars])
            continue
        added = len(para) + (2 if current else 0)
        if current and current_len + added > max_chars:
            emit()
        current.append(para)
        current_len += added if current_len else len(para)
    emit()
    return pieces or [body]
