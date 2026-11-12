import threading
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self._lock = threading.Lock()
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []
        self._doc_ranges: dict[str, tuple[int, int]] = {}
        self._bm25: BM25Okapi | None = None

    def add_documents(self, doc_id: str, chunks) -> None:
        with self._lock:
            start = len(self._corpus)
            for c in chunks:
                self._corpus.append(c.text)
                self._chunk_ids.append(f"{doc_id}_{c.index}")
                self._metadatas.append(c.metadata)
            end = len(self._corpus)
            self._doc_ranges[doc_id] = (start, end)
            self._rebuild()

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            if doc_id not in self._doc_ranges:
                return
            start, end = self._doc_ranges[doc_id]
            del self._corpus[start:end]
            del self._chunk_ids[start:end]
            del self._metadatas[start:end]
            del self._doc_ranges[doc_id]
            new_ranges = {}
            pos = 0
            for did, (s, e) in sorted(self._doc_ranges.items(), key=lambda x: x[1][0]):
                length = e - s
                new_ranges[did] = (pos, pos + length)
                pos += length
            self._doc_ranges = new_ranges
            self._rebuild()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        with self._lock:
            if not self._bm25 or not self._corpus:
                return []
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for i in top_indices:
                if scores[i] > 0:
                    results.append({
                        "id": self._chunk_ids[i],
                        "text": self._corpus[i],
                        "score": float(scores[i]),
                        "metadata": self._metadatas[i],
                    })
            return results

    def _rebuild(self):
        if self._corpus:
            tokenized = [doc.lower().split() for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def rebuild_from_chroma(self, collection) -> None:
        with self._lock:
            self._corpus.clear()
            self._chunk_ids.clear()
            self._metadatas.clear()
            self._doc_ranges.clear()

            results = collection.get(include=["documents", "metadatas"])
            if not results["ids"]:
                self._bm25 = None
                return

            doc_chunks: dict[str, list[tuple[str, str, dict]]] = {}
            for cid, text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                did = meta.get("doc_id", "")
                doc_chunks.setdefault(did, []).append((cid, text, meta))

            for did, items in doc_chunks.items():
                start = len(self._corpus)
                for cid, text, meta in items:
                    self._corpus.append(text)
                    self._chunk_ids.append(cid)
                    self._metadatas.append(meta)
                self._doc_ranges[did] = (start, len(self._corpus))

            self._rebuild()


bm25_index = BM25Index()
