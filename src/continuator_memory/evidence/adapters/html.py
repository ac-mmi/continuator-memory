"""HTML FormatAdapter: heading-first structure → CanonicalDocument.

Uses BeautifulSoup with the stdlib ``html.parser`` backend (no lxml).

Primary structure signal: ``h1``–``h6`` (same stack rules as Markdown).
Readable body: ``p`` and ``li``. Noise tags (``script`` / ``style`` /
``noscript`` / ``template``) are dropped. Semantic ``article`` / ``main`` /
``section`` may select a content root but do not invent heading hierarchy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from continuator_memory.evidence.canonical import CanonicalDocument, CanonicalSection
from continuator_memory.evidence.markdown import section_id_for
from continuator_memory.evidence.models import SOURCE_TYPE_HTML

_HTML_SUFFIXES = (".html", ".htm", ".xhtml")
_NOISE_TAGS = frozenset({"script", "style", "noscript", "template"})
_HEADING_TAGS = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}
_BODY_TAGS = frozenset({"p", "li"})
# Prefer document-like roots; do not use arbitrary nested <section> as the root.
_CONTENT_ROOT_TAGS = ("article", "main", "body")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _element_text(tag: Tag) -> str:
    return _normalize_text(tag.get_text(" ", strip=True))


def _parent_section_id(heading_path: tuple[str, ...]) -> str | None:
    if len(heading_path) <= 1:
        return None
    return section_id_for(heading_path[:-1])


@dataclass
class _SectionBuilder:
    heading_level: int
    heading_path: tuple[str, ...]
    title: str | None
    body_parts: list[str] = field(default_factory=list)
    children: list["_SectionBuilder"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_body(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            self.body_parts.append(cleaned)

    def to_canonical(self) -> CanonicalSection | None:
        text = "\n\n".join(self.body_parts).strip()
        children = tuple(
            child
            for child in (c.to_canonical() for c in self.children)
            if child is not None
        )
        if not text and not children:
            return None
        path = self.heading_path
        return CanonicalSection(
            section_id=section_id_for(path),
            heading_path=path,
            text=text,
            heading_level=self.heading_level,
            title=self.title,
            char_start=None,
            char_end=None,
            parent_section_id=_parent_section_id(path),
            children=children,
            metadata=dict(self.metadata),
        )


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag_name in _NOISE_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()


def _choose_content_root(soup: BeautifulSoup) -> tuple[Tag, str]:
    """Prefer article/main/section when present; otherwise body/html/soup."""
    for tag_name in _CONTENT_ROOT_TAGS:
        found = soup.find(tag_name)
        if found is not None and isinstance(found, Tag):
            return found, tag_name
    if soup.body is not None:
        return soup.body, "body"
    return soup, "document"


def _title_from_html(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.string:
        title = _normalize_text(str(soup.title.string))
        if title:
            return title
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        title = _element_text(h1)
        if title:
            return title
    return fallback


def _is_inside_heading(tag: Tag) -> bool:
    for parent in tag.parents:
        if isinstance(parent, Tag) and parent.name in _HEADING_TAGS:
            return True
    return False


def _iter_structure_events(root: Tag):
    """Yield ('heading', level, title, tag) or ('body', text, tag) in document order."""
    for element in root.descendants:
        if isinstance(element, NavigableString):
            continue
        if not isinstance(element, Tag):
            continue
        name = (element.name or "").lower()
        if name in _NOISE_TAGS:
            continue
        if name in _HEADING_TAGS:
            title = _element_text(element)
            if title:
                yield ("heading", _HEADING_TAGS[name], title, element)
            continue
        if name in _BODY_TAGS:
            if _is_inside_heading(element):
                continue
            text = _element_text(element)
            if text:
                yield ("body", text, element)


def html_to_canonical_sections(html: str) -> tuple[tuple[CanonicalSection, ...], dict[str, Any]]:
    """Parse HTML into nested CanonicalSections + adapter parse metadata."""
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    root, content_root = _choose_content_root(soup)

    roots: list[_SectionBuilder] = []
    stack: list[_SectionBuilder] = []
    preamble = _SectionBuilder(
        heading_level=0,
        heading_path=(),
        title=None,
        metadata={"html_role": "preamble"},
    )

    def current_builder() -> _SectionBuilder:
        return stack[-1] if stack else preamble

    for event in _iter_structure_events(root):
        if event[0] == "body":
            _, text, _tag = event
            current_builder().add_body(text)
            continue

        _, level, title, tag = event
        # Close open sections at this level or deeper (Markdown-equivalent stack).
        while stack and stack[-1].heading_level >= level:
            stack.pop()
        path = (stack[-1].heading_path + (title,)) if stack else (title,)
        node = _SectionBuilder(
            heading_level=level,
            heading_path=path,
            title=title,
            metadata={
                "html_tag": tag.name,
                "html_heading_level": level,
            },
        )
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    sections: list[CanonicalSection] = []
    preamble_section = preamble.to_canonical()
    if preamble_section is not None:
        sections.append(preamble_section)
    for root_builder in roots:
        canonical = root_builder.to_canonical()
        if canonical is not None:
            sections.append(canonical)

    parse_meta = {
        "content_root": content_root,
        "parser": "beautifulsoup4:html.parser",
    }
    return tuple(sections), parse_meta


class HTMLAdapter:
    """Parse HTML into CanonicalDocument."""

    name = "html"
    source_types = frozenset({SOURCE_TYPE_HTML})

    def supports(
        self,
        path_or_uri: str | Path,
        *,
        sniff: bytes | str | None = None,
    ) -> bool:
        text = str(path_or_uri).lower().split("?", 1)[0]
        if any(text.endswith(suffix) for suffix in _HTML_SUFFIXES):
            return True
        if sniff is None:
            return False
        sample = sniff.decode("utf-8", errors="replace") if isinstance(sniff, bytes) else sniff
        sample_l = sample.lstrip().lower()
        return sample_l.startswith("<!doctype html") or sample_l.startswith("<html") or bool(
            re.search(r"<h[1-6]\b", sample_l)
        )

    def parse(
        self,
        source: str | bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
        title_fallback: str | None = None,
    ) -> CanonicalDocument:
        if isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            text = source
        meta = dict(metadata or {})
        fallback = title_fallback
        if fallback is None:
            original = meta.get("original_path")
            if original:
                fallback = Path(str(original)).stem
            else:
                fallback = Path(source_uri).stem or "document"

        soup = BeautifulSoup(text, "html.parser")
        title = _title_from_html(soup, fallback=str(fallback))
        sections, parse_meta = html_to_canonical_sections(text)
        meta = {**meta, **parse_meta}
        if soup.title and soup.title.string:
            html_title = _normalize_text(str(soup.title.string))
            if html_title:
                meta.setdefault("html_title", html_title)

        return CanonicalDocument(
            source_uri=source_uri,
            source_type=SOURCE_TYPE_HTML,
            content_hash=_content_hash(text),
            title=title,
            metadata=meta,
            sections=sections,
        )
