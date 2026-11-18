import io
from dataclasses import dataclass
from pathlib import PurePath


class ExtractionError(Exception):
    """The file cannot be turned into text (corrupt, encrypted, scanned, empty)."""


@dataclass
class Page:
    number: int | None  # None for formats without pages (txt, docx)
    text: str


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_pages(data: bytes, filename: str) -> list[Page]:
    ext = PurePath(filename).suffix.lower()
    if ext == ".pdf":
        pages = _extract_pdf(data)
    elif ext == ".docx":
        pages = _extract_docx(data)
    elif ext in {".txt", ".md"}:
        pages = [Page(None, data.decode("utf-8", errors="replace"))]
    else:
        raise ExtractionError(f"Unsupported file type: {ext or 'unknown'}")
    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise ExtractionError("No extractable text found (scanned PDFs need OCR, which is not supported).")
    return pages


def _extract_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("Encrypted PDFs are not supported.")
        return [Page(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except ExtractionError:
        raise
    except (PyPdfError, ValueError, KeyError) as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc


def _extract_docx(data: bytes) -> list[Page]:
    from docx import Document as DocxDocument

    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001  (python-docx raises assorted errors)
        raise ExtractionError(f"Could not read DOCX: {exc}") from exc
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [Page(None, "\n\n".join(parts))]
