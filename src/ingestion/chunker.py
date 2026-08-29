def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    source = pages[0]["source"] if pages else ""
    words: list[tuple[str, int]] = []
    for page in pages:
        words.extend((w, page["page"]) for w in page["text"].split())

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for chunk_index, i in enumerate(range(0, len(words), step)):
        window = words[i : i + chunk_size]
        if not window:
            break
        chunk_pages = sorted({p for _, p in window})
        chunks.append(
            {
                "chunk_id": f"{source}::chunk_{chunk_index}",
                "text": " ".join(w for w, _ in window),
                "source": source,
                "page": chunk_pages[0],
                "page_range": chunk_pages,
                "chunk_index": chunk_index,
            }
        )
        if i + chunk_size >= len(words):
            break
    return chunks
