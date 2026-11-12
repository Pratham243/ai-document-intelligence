import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.retrieval.bm25_index import bm25_index
from app.retrieval.vector_store import get_chroma_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)

    try:
        collection = get_chroma_collection()
        bm25_index.rebuild_from_chroma(collection)
    except Exception:
        pass
    yield


app = FastAPI(
    title="AI Document Intelligence",
    description="Production RAG pipeline with hybrid search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)
