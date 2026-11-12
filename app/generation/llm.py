import httpx
from groq import Groq
from app.config import settings
from app.observability.langfuse_client import trace_llm_call


SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question based ONLY on the provided context chunks.
If the context doesn't contain enough information, say so honestly.
Always cite which chunk(s) you used by referencing the source filename.
Be concise and accurate."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        src = chunk.get("metadata", {}).get("filename", "unknown")
        context_parts.append(f"[Chunk {i+1} | Source: {src}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    return f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite sources):"""


def generate_with_groq(query: str, chunks: list[dict], trace_id: str | None = None) -> dict:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=settings.groq_api_key)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    trace_llm_call(
        trace_id=trace_id,
        model=settings.groq_model,
        prompt=prompt,
        response=answer,
        provider="groq",
    )
    return {"answer": answer, "provider": "groq", "model": settings.groq_model}


def generate_with_ollama(query: str, chunks: list[dict], trace_id: str | None = None) -> dict:
    prompt = build_prompt(query, chunks)
    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("message", {}).get("content", "")

    trace_llm_call(
        trace_id=trace_id,
        model=settings.ollama_model,
        prompt=prompt,
        response=answer,
        provider="ollama",
    )
    return {"answer": answer, "provider": "ollama", "model": settings.ollama_model}


def generate_no_llm(query: str, chunks: list[dict]) -> dict:
    parts = []
    for i, chunk in enumerate(chunks):
        src = chunk.get("metadata", {}).get("filename", "unknown")
        parts.append(f"**[Source: {src}]**\n{chunk['text']}")
    return {
        "answer": "**LLM unavailable — showing retrieved chunks verbatim:**\n\n" + "\n\n---\n\n".join(parts),
        "provider": "none",
        "model": "none",
    }


def generate_answer(query: str, chunks: list[dict], mode: str = "groq", trace_id: str | None = None) -> dict:
    if mode == "groq":
        try:
            return generate_with_groq(query, chunks, trace_id=trace_id)
        except Exception:
            pass
        try:
            return generate_with_ollama(query, chunks, trace_id=trace_id)
        except Exception:
            pass
        return generate_no_llm(query, chunks)
    elif mode == "ollama":
        try:
            return generate_with_ollama(query, chunks, trace_id=trace_id)
        except Exception:
            pass
        return generate_no_llm(query, chunks)
    else:
        return generate_no_llm(query, chunks)
