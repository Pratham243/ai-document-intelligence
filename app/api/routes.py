import os
import json
import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from app.config import settings
from app.ingestion.pipeline import ingest_document, list_documents, delete_document
from app.retrieval.hybrid import hybrid_search
from app.generation.llm import generate_answer
from app.safety.guardrails import check_prompt_injection
from app.observability.langfuse_client import create_trace, trace_retrieval, get_trace_url

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    mode: str = "groq"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    provider: str
    model: str
    trace_url: Optional[str] = None
    latency_ms: float


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

    content = await file.read()
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}{ext}")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_document(file_path=file_path, filename=file.filename)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@router.get("/documents")
async def get_documents():
    return list_documents()


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(404, "Document not found")
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    start = time.time()

    trace_id = create_trace(name="rag-query", metadata={"question": req.question, "mode": req.mode})

    blocked, msg = check_prompt_injection(req.question, trace_id=trace_id)
    if blocked:
        raise HTTPException(400, msg)

    results = hybrid_search(req.question)
    trace_retrieval(trace_id, req.question, results)

    if not results:
        return QueryResponse(
            answer="No relevant documents found. Please upload documents first.",
            sources=[],
            provider="none",
            model="none",
            trace_url=get_trace_url(trace_id),
            latency_ms=(time.time() - start) * 1000,
        )

    gen = generate_answer(req.question, results, mode=req.mode, trace_id=trace_id)

    sources = []
    for r in results:
        sources.append({
            "filename": r.get("metadata", {}).get("filename", ""),
            "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
            "text": r.get("text", "")[:300],
            "score": r.get("rerank_score", r.get("score", 0)),
        })

    latency = (time.time() - start) * 1000
    return QueryResponse(
        answer=gen["answer"],
        sources=sources,
        provider=gen["provider"],
        model=gen["model"],
        trace_url=get_trace_url(trace_id),
        latency_ms=latency,
    )


@router.post("/query/voice")
async def query_voice(audio: UploadFile = File(...), mode: str = Form("groq")):
    content = await audio.read()

    try:
        from app.voice.whisper import transcribe_audio
        question = transcribe_audio(content, filename=audio.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    if not question.strip():
        raise HTTPException(400, "Could not transcribe audio")

    start = time.time()
    trace_id = create_trace(name="voice-query", metadata={"question": question, "mode": mode})

    blocked, msg = check_prompt_injection(question, trace_id=trace_id)
    if blocked:
        raise HTTPException(400, msg)

    results = hybrid_search(question)
    trace_retrieval(trace_id, question, results)

    if not results:
        return {
            "transcription": question,
            "answer": "No relevant documents found.",
            "sources": [],
            "provider": "none",
            "model": "none",
            "trace_url": get_trace_url(trace_id),
            "latency_ms": (time.time() - start) * 1000,
        }

    gen = generate_answer(question, results, mode=mode, trace_id=trace_id)

    sources = [{
        "filename": r.get("metadata", {}).get("filename", ""),
        "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
        "text": r.get("text", "")[:300],
        "score": r.get("rerank_score", r.get("score", 0)),
    } for r in results]

    return {
        "transcription": question,
        "answer": gen["answer"],
        "sources": sources,
        "provider": gen["provider"],
        "model": gen["model"],
        "trace_url": get_trace_url(trace_id),
        "latency_ms": (time.time() - start) * 1000,
    }


@router.get("/health")
async def health():
    from app.retrieval.vector_store import get_chroma_collection
    try:
        collection = get_chroma_collection()
        count = collection.count()
    except Exception:
        count = -1
    return {
        "status": "healthy",
        "chroma_documents": count,
        "llm_mode": settings.llm_mode,
    }


@router.get("/evaluation/results")
async def evaluation_results():
    results_path = Path("tests/evaluation/results.json")
    if not results_path.exists():
        raise HTTPException(404, "Evaluation results not found. Run: pytest tests/evaluation/")
    return json.loads(results_path.read_text())
