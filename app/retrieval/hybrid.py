from functools import lru_cache
from sentence_transformers import CrossEncoder
from app.config import settings
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import get_chroma_collection
from app.retrieval.bm25_index import bm25_index


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs[doc_id] = item
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs[did] for did in sorted_ids]


def hybrid_search(query: str) -> list[dict]:
    bm25_results = bm25_index.search(query, top_k=settings.bm25_top_k)

    embed_model = get_embedding_model()
    query_embedding = embed_model.encode(query).tolist()
    collection = get_chroma_collection()

    n_docs = collection.count()
    if n_docs == 0:
        return []

    sem_k = min(settings.semantic_top_k, n_docs)
    sem_results_raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=sem_k,
        include=["documents", "metadatas", "distances"],
    )

    semantic_results = []
    if sem_results_raw["ids"] and sem_results_raw["ids"][0]:
        for i, cid in enumerate(sem_results_raw["ids"][0]):
            semantic_results.append({
                "id": cid,
                "text": sem_results_raw["documents"][0][i],
                "score": 1.0 - sem_results_raw["distances"][0][i],
                "metadata": sem_results_raw["metadatas"][0][i],
            })

    fused = reciprocal_rank_fusion([bm25_results, semantic_results], k=settings.rrf_k)

    candidates = fused[:20]
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:settings.rerank_top_k]
