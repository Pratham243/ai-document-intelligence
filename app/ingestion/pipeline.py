import os
import uuid

from app.config import settings
from app.ingestion.extractors import extract_text
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import get_chroma_collection
from app.retrieval.bm25_index import bm25_index


def ingest_document(
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    filename: str = "",
) -> dict:
    doc_id = str(uuid.uuid4())
    actual_filename = filename or (os.path.basename(file_path) if file_path else "unknown")

    text = extract_text(file_path=file_path, file_bytes=file_bytes, filename=actual_filename)
    if not text.strip():
        raise ValueError("No text extracted from document")

    chunks = chunk_text(
        text, doc_id, actual_filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embed_model = get_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

    collection = get_chroma_collection()
    ids = [f"{doc_id}_{c.index}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    bm25_index.add_documents(doc_id, chunks)

    return {
        "doc_id": doc_id,
        "filename": actual_filename,
        "num_chunks": len(chunks),
        "text_length": len(text),
    }


def list_documents() -> list[dict]:
    collection = get_chroma_collection()
    results = collection.get(include=["metadatas"])
    docs = {}
    for meta in results["metadatas"]:
        did = meta.get("doc_id", "")
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "filename": meta.get("filename", ""),
                "num_chunks": 0,
            }
        docs[did]["num_chunks"] += 1
    return list(docs.values())


def delete_document(doc_id: str) -> bool:
    collection = get_chroma_collection()
    results = collection.get(where={"doc_id": doc_id}, include=[])
    if not results["ids"]:
        return False
    collection.delete(ids=results["ids"])
    bm25_index.remove_document(doc_id)
    return True
