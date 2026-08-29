from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.chunker import chunk_text
from src.ingestion.parser import parse_document
from src.retrieval.bm25 import rebuild_bm25_from_vectorstore
from src.retrieval.embedder import embed_texts
from src.retrieval.vectorstore import add_documents
from src.security.pii import mask_pii


@dataclass(frozen=True)
class IngestResult:
    source: str
    pages: int
    chunks: int
    chunk_ids: list[str] = field(default_factory=list)


def ingest_document(
    file_path: str | Path,
    *,
    chunk_size: int = 500,
    overlap: int = 100,
    mask: bool = True,
) -> IngestResult:
    """Parse → chunk → mask PII → embed → upsert to the vector store → rebuild BM25.

    PII masking runs before embedding and storage on purpose: once a chunk is
    embedded and persisted, the vector encodes whatever PII the chunk held, and
    redacting the source text afterwards does not undo that. In-flight masking
    is the only point where the vector store can be kept clean.

    Re-ingesting the same file is idempotent — chunk ids are deterministic and
    the store upserts — so this is safe to call again after a document changes.
    """
    path = Path(file_path)
    pages = parse_document(path)
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return IngestResult(source=path.name, pages=len(pages), chunks=0, chunk_ids=[])

    texts = [mask_pii(chunk["text"]) if mask else chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            # Chroma metadata values must be scalar (str/int/float/bool),
            # so the page list is flattened to "1,2,3".
            "page_range": ",".join(str(page) for page in chunk["page_range"]),
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]
    embeddings = embed_texts(texts)
    add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    # BM25 is in-memory and is rebuilt by re-reading the whole collection, so
    # it must run after the upsert above for the new chunks to be included.
    rebuild_bm25_from_vectorstore()

    return IngestResult(
        source=path.name,
        pages=len(pages),
        chunks=len(chunks),
        chunk_ids=ids,
    )
