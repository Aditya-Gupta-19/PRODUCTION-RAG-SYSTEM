from pathlib import Path

from pypdf import PdfReader


def parse_pdf(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    reader = PdfReader(str(path))
    return [
        {"page": i + 1, "text": page.extract_text() or "", "source": path.name}
        for i, page in enumerate(reader.pages)
    ]


_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def parse_txt(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    return [{"page": 1, "text": path.read_text(encoding="utf-8"), "source": path.name}]


def parse_document(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in _TEXT_SUFFIXES:
        return parse_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")
