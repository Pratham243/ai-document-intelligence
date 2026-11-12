from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "documents"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    bm25_top_k: int = 10
    semantic_top_k: int = 10
    rerank_top_k: int = 3
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_mode: str = "groq"

    # Whisper
    whisper_model: str = "base.en"

    # LangFuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Data
    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"
    sample_docs_dir: str = "./data/sample_docs"


settings = Settings()
