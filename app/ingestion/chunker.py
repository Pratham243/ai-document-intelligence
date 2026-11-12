import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict


def estimate_tokens(text: str) -> int:
    return len(text.split())


def split_into_sentences(text: str) -> list[str]:
    return re.split(r'(?<=[.!?])\s+', text)


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                index=idx,
                metadata={"doc_id": doc_id, "filename": filename, "chunk_index": idx},
            ))
            idx += 1

            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(Chunk(
            text=" ".join(current_sentences),
            index=idx,
            metadata={"doc_id": doc_id, "filename": filename, "chunk_index": idx},
        ))

    return chunks
