"""Format adapters that parse sources into CanonicalDocument.

Implemented: Markdown, HTML, DOCX, PDF (digital text).
"""

from continuator_memory.evidence.adapters.docx import DOCXAdapter
from continuator_memory.evidence.adapters.html import HTMLAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.adapters.pdf import PDFAdapter

__all__ = ["DOCXAdapter", "HTMLAdapter", "MarkdownAdapter", "PDFAdapter"]
