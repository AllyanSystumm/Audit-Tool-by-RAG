"""Configuration management for the Audit Tool."""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _parse_list(value: str) -> list:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
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

    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_INITIAL_RETRY_DELAY = float(os.getenv("LLM_INITIAL_RETRY_DELAY", "1.0"))
    LLM_MAX_RETRY_DELAY = float(os.getenv("LLM_MAX_RETRY_DELAY", "60.0"))
    LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120.0"))
    LLM_HEALTH_CHECK_INTERVAL = int(os.getenv("LLM_HEALTH_CHECK_INTERVAL", "300"))
    LLM_FALLBACK_MODELS = _parse_list(os.getenv("LLM_FALLBACK_MODELS", ""))

    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx", "xls", "xml", "pptx", "txt"}
    FILENAME_MAX_LENGTH = int(os.getenv("FILENAME_MAX_LENGTH", "255"))

    API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() in ("1", "true", "yes", "y")
    API_KEYS = set(_parse_list(os.getenv("API_KEYS", "")))
    API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() in ("1", "true", "yes", "y")
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))

    EXTERNAL_REQUEST_TIMEOUT = float(os.getenv("EXTERNAL_REQUEST_TIMEOUT", "30.0"))

    STRUCTURED_LOGGING = os.getenv("STRUCTURED_LOGGING", "false").lower() in ("1", "true", "yes", "y")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls):
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_filename(cls, filename: str) -> tuple:
        if not filename:
            return False, "Filename is empty"
        
        if len(filename) > cls.FILENAME_MAX_LENGTH:
            return False, f"Filename exceeds maximum length of {cls.FILENAME_MAX_LENGTH}"
        
        dangerous_patterns = [
            r"\.\.",
            r"^/",
            r"^\\",
            r"^~",
            r"[<>:\"|?*]",
            r"[\x00-\x1f]",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, filename):
                return False, f"Filename contains invalid characters"
        
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"File extension '.{ext}' is not allowed. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
        
        return True, None

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        filename = filename.replace("..", "_")
        filename = filename.lstrip("/\\~")
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", filename)
        
        name = Path(filename).stem
        ext = Path(filename).suffix
        
        if not ext and name.startswith("."):
            ext = name
            name = ""
        
        name = re.sub(r"[^\w\-. ]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_. ")
        
        if not name:
            name = "unnamed"
        
        max_name_len = cls.FILENAME_MAX_LENGTH - len(ext) - 1
        if len(name) > max_name_len:
            name = name[:max_name_len]
        
        return f"{name}{ext}"


config = Config()
config.ensure_directories()
