import uuid
from app.config import settings

_langfuse = None


def get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return _langfuse
    except Exception:
        return None


def create_trace(name: str = "rag-query", metadata: dict | None = None) -> str | None:
    lf = get_langfuse()
    if lf is None:
        return None
    trace_id = str(uuid.uuid4())
    try:
        lf.trace(id=trace_id, name=name, metadata=metadata or {})
        return trace_id
    except Exception:
        return None


def trace_retrieval(trace_id: str | None, query: str, results: list[dict]) -> None:
    if trace_id is None:
        return
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.span(
            trace_id=trace_id,
            name="retrieval",
            input={"query": query},
            output={
                "num_results": len(results),
                "top_sources": [
                    r.get("metadata", {}).get("filename", "") for r in results[:3]
                ],
            },
        )
    except Exception:
        pass


def trace_llm_call(
    trace_id: str | None, model: str, prompt: str, response: str, provider: str
) -> None:
    if trace_id is None:
        return
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.generation(
            trace_id=trace_id,
            name="llm-generation",
            model=model,
            input=prompt,
            output=response,
            metadata={"provider": provider},
        )
    except Exception:
        pass


def trace_guardrail(trace_id: str | None, query: str, blocked: bool) -> None:
    if trace_id is None:
        return
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.span(
            trace_id=trace_id,
            name="guardrail-check",
            input={"query": query},
            output={"blocked": blocked},
        )
    except Exception:
        pass


def get_trace_url(trace_id: str | None) -> str | None:
    if trace_id is None:
        return None
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        return f"{settings.langfuse_host}/trace/{trace_id}"
    except Exception:
        return None
