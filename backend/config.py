"""Configuration management for the Audit Tool."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    _raw_vector_db_path = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "vector_db"))
    VECTOR_DB_PATH = str((BASE_DIR / _raw_vector_db_path).resolve()) if not Path(_raw_vector_db_path).is_absolute() else _raw_vector_db_path

    HF_TOKEN = os.getenv("HF_TOKEN", "")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    EMBEDDING_CACHE_MAX = int(os.getenv("EMBEDDING_CACHE_MAX", "2048"))

    ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("1", "true", "yes", "y")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "50"))  

    PROFILE_LOCK_STANDARD_CLAUSE = os.getenv("PROFILE_LOCK_STANDARD_CLAUSE", "true").lower() in ("1", "true", "yes", "y")

    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    HNSW_M = int(os.getenv("HNSW_M", "32"))
    HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "128"))

    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "15"))
    HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

    BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

    NUM_CHECKPOINTS = 5

    KEEP_UPLOADED_FILES = os.getenv("KEEP_UPLOADED_FILES", "false").lower() in ("1", "true", "yes", "y")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_directories()

