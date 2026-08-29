import pytest
from fpdf import FPDF

from src.ingestion.parser import parse_document, parse_pdf, parse_txt


def test_parse_txt_returns_single_page(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("Hello from a text file.", encoding="utf-8")

    pages = parse_txt(file_path)

    assert pages == [{"page": 1, "text": "Hello from a text file.", "source": "note.txt"}]


def test_parse_pdf_extracts_text_per_page(tmp_path):
    file_path = tmp_path / "sample.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Hello world, this is a test PDF.")
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Second page content.")
    pdf.output(str(file_path))

    pages = parse_pdf(file_path)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Hello world" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "Second page" in pages[1]["text"]
    assert all(p["source"] == "sample.pdf" for p in pages)


def test_parse_document_dispatches_by_suffix(tmp_path):
    txt_path = tmp_path / "a.txt"
    txt_path.write_text("hi", encoding="utf-8")
    assert parse_document(txt_path) == parse_txt(txt_path)


def test_parse_document_treats_markdown_as_text(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Heading\n\nBody text.", encoding="utf-8")
    pages = parse_document(md_path)
    assert pages == [{"page": 1, "text": "# Heading\n\nBody text.", "source": "notes.md"}]


def test_parse_document_rejects_unsupported_type(tmp_path):
    bad_path = tmp_path / "a.docx"
    bad_path.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_document(bad_path)
