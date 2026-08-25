import pytest

from src.ingestion.chunker import chunk_text


def _pages(word_count: int, source: str = "doc.txt", page: int = 1) -> list[dict]:
    text = " ".join(f"word{i}" for i in range(word_count))
    return [{"page": page, "text": text, "source": source}]


def test_single_chunk_when_under_chunk_size():
    pages = _pages(300)
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0]["text"].split() == [f"word{i}" for i in range(300)]
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["page"] == 1


def test_overlap_between_consecutive_chunks():
    pages = _pages(1000)
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 3  # windows: [0:500], [400:900], [800:1000]

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert len(first_words) == 500
    assert len(second_words) == 500
    assert set(first_words[-100:]) == set(second_words[:100])
    assert len(chunks[2]["text"].split()) == 200  # tail chunk shorter than chunk_size


def test_exact_boundary_produces_no_short_tail():
    pages = _pages(1300)  # windows [0,500) [400,900) [800,1300) land exactly on the end
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert all(len(c["text"].split()) == 500 for c in chunks)


def test_empty_pages_returns_no_chunks():
    assert chunk_text([], chunk_size=500, overlap=100) == []
    assert chunk_text([{"page": 1, "text": "", "source": "x.txt"}]) == []


def test_chunk_ids_are_unique_and_ordered():
    pages = _pages(1000)
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_spans_multiple_pages_reports_page_range():
    pages = [
        {"page": 1, "text": " ".join(f"a{i}" for i in range(400)), "source": "doc.txt"},
        {"page": 2, "text": " ".join(f"b{i}" for i in range(400)), "source": "doc.txt"},
    ]
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    assert chunks[0]["page_range"] == [1, 2]
    assert chunks[0]["page"] == 1


def test_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(_pages(10), chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text(_pages(10), chunk_size=100, overlap=150)
