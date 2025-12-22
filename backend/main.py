"""FastAPI backend for Audit Tool RAG application."""

import os
import sys
from pathlib import Path
from typing import Optional
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import config
from backend.ingestion.document_processor import DocumentProcessor
from backend.ingestion.chunker import SemanticChunker
from backend.ingestion.embedder import EmbeddingGenerator
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_search import HybridRetriever
from backend.generation.llm_client import LLMClient, LLMClientFactory
from backend.generation.checkpoint_generator import CheckpointGenerator
from backend.generation.process_profiles import detect_process_type
from backend.generation.prompts import _extract_headings_and_excerpts
from backend.middleware.auth import APIKeyMiddleware
from backend.middleware.rate_limiter import RateLimiterMiddleware, TokenBucketRateLimiter
from backend.middleware.logging_utils import (
    RequestLoggingMiddleware,
    get_logger,
    get_metrics_collector
)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = get_logger(__name__)

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class FileSizeError(Exception):
    pass


class FilenameError(Exception):
    pass


async def validate_upload(file: UploadFile) -> tuple:
    is_valid, error = config.validate_filename(file.filename)
    if not is_valid:
        raise FilenameError(error)
    
    content = await file.read()
    await file.seek(0)
    
    if len(content) > config.MAX_UPLOAD_SIZE_BYTES:
        raise FileSizeError(
            f"File size ({len(content) / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({config.MAX_UPLOAD_SIZE_MB}MB)"
        )
    
    safe_filename = config.sanitize_filename(file.filename)
    
    return content, safe_filename


def _cleanup_uploaded_file(path: Path):
    try:
        if config.KEEP_UPLOADED_FILES:
            logger.info("KEEP_UPLOADED_FILES=true; keeping uploaded file", path=str(path))
            return
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as e:
        logger.exception("Failed to clean up uploaded file", path=str(path), error=str(e))


def _rebuild_bm25_from_vector_store() -> int:
    global bm25_retriever, vector_store
    if not bm25_retriever or not vector_store:
        return 0

    try:
        all_docs = vector_store.get_all_documents()
        docs = all_docs.get("documents") or []
        metas = all_docs.get("metadatas") or []
        ids = all_docs.get("ids") or []

        chunks = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            doc_id = ids[i] if i < len(ids) else None
            chunk = {"text": text, "metadata": meta}
            if doc_id:
                chunk["id"] = doc_id
            chunks.append(chunk)

        bm25_retriever.index_documents(chunks)
        return len(chunks)
    except Exception:
        logger.exception("Failed to rebuild BM25 from vector store")
        return 0


def _merge_results(results_lists, top_k: int):
    merged = {}
    for results in results_lists:
        for r in results or []:
            rid = r.get("id") or r.get("text", "")[:100]
            if not rid:
                continue
            if rid not in merged or r.get("hybrid_score", 0) > merged[rid].get("hybrid_score", 0):
                merged[rid] = r
    ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)

    max_per_source = 4
    per_source = {}
    diversified = []
    for r in ranked:
        md = r.get("metadata") or {}
        source = md.get("filename") or md.get("source") or "Unknown"
        per_source[source] = per_source.get(source, 0) + 1
        if per_source[source] > max_per_source:
            continue
        diversified.append(r)
        if len(diversified) >= top_k:
            break
    return diversified[:top_k]


app = FastAPI(
    title="Audit Tool API",
    description="RAG-powered checkpoint generation for audit processes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

rate_limiter = TokenBucketRateLimiter()
app.add_middleware(RateLimiterMiddleware, limiter=rate_limiter)

app.add_middleware(APIKeyMiddleware)

document_processor = None
chunker = None
embedder = None
vector_store = None
bm25_retriever = None
hybrid_retriever = None
llm_client = None
checkpoint_generator = None


@app.on_event("startup")
async def startup_event():
    global document_processor, chunker, embedder, vector_store
    global bm25_retriever, hybrid_retriever, llm_client, checkpoint_generator
    
    logger.info("Initializing Audit Tool components...")
    
    try:
        document_processor = DocumentProcessor()
        chunker = SemanticChunker()
        embedder = EmbeddingGenerator()
        vector_store = VectorStore()
        bm25_retriever = BM25Retriever()
        hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)

        existing_count = vector_store.get_count()
        if existing_count > 0:
            rebuilt = _rebuild_bm25_from_vector_store()
            logger.info("Rebuilt BM25 index from Chroma", count=rebuilt)
        
        if config.HF_TOKEN:
            llm_client = LLMClient()
            checkpoint_generator = CheckpointGenerator(llm_client)
            logger.info("LLM client initialized successfully")
        else:
            logger.warning("HF_TOKEN not set. LLM features will be disabled.")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error("Error initializing components", error=str(e))


@app.get("/")
def read_root():
    return {
        "message": "Audit Tool API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "generate": "/generate-checkpoints",
            "ingest": "/ingest-documents"
        }
    }


@app.get("/health")
def health_check():
    llm_health = None
    if llm_client:
        try:
            llm_health = llm_client.get_health_status()
        except Exception:
            llm_health = {"healthy": False, "error": "Unable to get status"}

    return {
        "status": "healthy",
        "components": {
            "document_processor": document_processor is not None,
            "embedder": embedder is not None,
            "vector_store": vector_store is not None,
            "llm_client": llm_client is not None,
            "vector_db_count": vector_store.get_count() if vector_store else 0,
            "llm_health": llm_health
        },
        "config": {
            "max_upload_size_mb": config.MAX_UPLOAD_SIZE_MB,
            "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
            "api_key_enabled": config.API_KEY_ENABLED
        }
    }


@app.get("/health/llm")
def llm_health_check(force: bool = False):
    if not llm_client:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialized. Please set HF_TOKEN environment variable."
        )
    
    try:
        is_healthy = llm_client.check_health(force=force)
        status = llm_client.get_health_status()
        
        if not is_healthy:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "details": status}
            )
        
        return {"status": "healthy", "details": status}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)}
        )


@app.get("/metrics")
def get_metrics():
    collector = get_metrics_collector()
    stats = collector.get_stats()
    
    if embedder:
        stats["embedding_cache"] = embedder.get_cache_stats()
    
    if vector_store:
        stats["vector_store_count"] = vector_store.get_count()
    
    if bm25_retriever:
        stats["bm25_count"] = bm25_retriever.get_document_count()
    
    return stats


@app.post("/ingest-documents")
async def ingest_documents(file: UploadFile = File(...)):
    if not all([document_processor, chunker, embedder, vector_store, bm25_retriever]):
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        content, safe_filename = await validate_upload(file)
    except FileSizeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FilenameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        file_path = config.UPLOADS_DIR / safe_filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info("Processing file for ingestion", filename=safe_filename)
        
        extraction_result = document_processor.extract_text(str(file_path))
        
        if extraction_result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text: {extraction_result.get('error', 'Unknown error')}"
            )
        
        text = extraction_result['text']
        metadata = extraction_result['metadata']

        ptype = detect_process_type(filename=safe_filename, text=text)
        metadata = {**(metadata or {}), "process_type": ptype}
        
        chunks = chunker.chunk_text(text, metadata)
        logger.info("Created chunks", count=len(chunks))
        
        chunks_with_embeddings = embedder.encode_chunks(chunks)
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        
        doc_id = hashlib.sha1(content).hexdigest()[:12]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, embeddings, ids=ids)
        
        bm25_count = _rebuild_bm25_from_vector_store()
        logger.info("BM25 index updated", count=bm25_count)
        
        _cleanup_uploaded_file(file_path)
        
        return {
            "status": "success",
            "filename": safe_filename,
            "chunks_created": len(chunks),
            "total_documents_in_db": vector_store.get_count(),
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error ingesting document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm-models")
def list_llm_models():
    return {
        "status": "success",
        "models": config.get_available_llms(),
        "default": config.DEFAULT_LLM,
        "current": config.LLM_MODEL
    }


@app.post("/generate-checkpoints")
async def generate_checkpoints(
    file: UploadFile = File(...),
    use_rag: bool = True,
    num_checkpoints: int = 5,
    ingest_to_kb: bool = False,
    process_type: str = "auto",
    llm_model: str = None
):
    selected_model = llm_model or config.DEFAULT_LLM
    
    try:
        selected_llm_client = LLMClientFactory.get_client(selected_model)
        selected_checkpoint_gen = CheckpointGenerator(selected_llm_client)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM model '{selected_model}': {str(e)}"
        )
    
    if not selected_checkpoint_gen:
        raise HTTPException(
            status_code=500,
            detail="LLM client not initialized. Please set HF_TOKEN environment variable."
        )
    
    if not all([document_processor, chunker, embedder, vector_store, bm25_retriever, hybrid_retriever]):
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        content, safe_filename = await validate_upload(file)
    except FileSizeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FilenameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        file_path = config.UPLOADS_DIR / safe_filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info("Generating checkpoints", filename=safe_filename)
        
        extraction_result = document_processor.extract_text(str(file_path))
        
        if extraction_result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text: {extraction_result.get('error', 'Unknown error')}"
            )
        
        process_document = extraction_result['text']
        metadata = extraction_result['metadata']

        resolved_ptype = process_type if process_type and process_type != "auto" else detect_process_type(
            filename=safe_filename,
            text=process_document
        )
        filter_metadata = {"process_type": resolved_ptype} if resolved_ptype and resolved_ptype != "auto" else None
        
        if ingest_to_kb:
            logger.info("Ingesting uploaded document into Knowledge Base...")
            ingest_md = {**(metadata or {}), "process_type": resolved_ptype}
            chunks = chunker.chunk_text(process_document, ingest_md)
            logger.info("Created chunks for ingestion", count=len(chunks))

            texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(texts)

            doc_id = hashlib.sha1(content).hexdigest()[:12]
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

            vector_store.add_documents(chunks, embeddings, ids=ids)
            logger.info("Stored chunks in vector database", count=len(chunks))

            bm25_count = _rebuild_bm25_from_vector_store()
            logger.info("BM25 index updated", count=bm25_count)
        
        relevant_chunks = []
        retrieved_chunks_preview = []
        if use_rag and vector_store.get_count() > 0:
            try:
                logger.info("Retrieving relevant context using hybrid search...")
                heading_pairs = _extract_headings_and_excerpts(process_document, max_items=6)
                queries = []
                for h, ex in heading_pairs[:5]:
                    queries.append(f"{h}\n{ex}")
                queries.append(process_document[:1800])
                if not queries:
                    queries = [process_document[:1800]]

                results_lists = [
                    hybrid_retriever.search(
                        q,
                        embedder,
                        top_k=config.TOP_K_RETRIEVAL,
                        filter_metadata=filter_metadata,
                        use_reranker=True
                    )
                    for q in queries
                ]
                relevant_chunks = _merge_results(results_lists, top_k=config.TOP_K_RETRIEVAL)
                logger.info("Retrieved relevant chunks", count=len(relevant_chunks))

                for r in relevant_chunks:
                    md = r.get("metadata") or {}
                    retrieved_chunks_preview.append({
                        "id": r.get("id"),
                        "score": r.get("hybrid_score", r.get("score")),
                        "source": md.get("filename") or md.get("source") or "Unknown",
                        "text_preview": (r.get("text") or "")[:350]
                    })
            except Exception:
                logger.exception("RAG retrieval failed")
                raise HTTPException(
                    status_code=500,
                    detail="RAG retrieval failed (vector/BM25 search). Check backend logs."
                )
        
        result = selected_checkpoint_gen.generate_checkpoints(
            process_document,
            relevant_chunks,
            num_checkpoints,
            process_type=process_type,
            filename=safe_filename
        )
        
        _cleanup_uploaded_file(file_path)
        
        if result['status'] != 'success':
            raise HTTPException(
                status_code=500,
                detail=f"Error generating checkpoints: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "status": "success",
            "filename": safe_filename,
            "checkpoints": result['checkpoints'],
            "num_checkpoints": result['num_checkpoints'],
            "used_rag": use_rag and len(relevant_chunks) > 0,
            "num_context_chunks": len(relevant_chunks),
            "ingest_to_kb": ingest_to_kb,
            "retrieved_chunks": retrieved_chunks_preview,
            "process_type": result.get("process_type", "auto"),
            "process_profile": result.get("process_profile"),
            "process_doc_score": result.get("process_doc_score"),
            "raw_output": result.get('raw_output', ''),
            "llm_model": selected_model,
            "llm_model_id": config.get_llm_model_id(selected_model)
        }
        
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error generating checkpoints")
        raise HTTPException(
            status_code=500,
            detail="Internal error while generating checkpoints. Check backend logs."
        )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not document_processor:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    try:
        content, safe_filename = await validate_upload(file)
    except FileSizeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FilenameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        file_path = config.UPLOADS_DIR / safe_filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        result = document_processor.extract_text(str(file_path))
        
        _cleanup_uploaded_file(file_path)
        
        if result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "status": "success",
            "filename": safe_filename,
            "text": result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
            "full_text_length": len(result['text']),
            "metadata": result['metadata']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading file", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-database")
async def reset_database():
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        vector_store.reset_collection()
        
        global bm25_retriever
        bm25_retriever = BM25Retriever()
        global hybrid_retriever
        hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
        
        return {
            "status": "success",
            "message": "Database reset successfully"
        }
        
    except Exception as e:
        logger.error("Error resetting database", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/database-info")
def get_database_info():
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        count = vector_store.get_count()
        bm25_count = bm25_retriever.get_document_count() if bm25_retriever else 0
        
        embedding_cache_stats = None
        if embedder:
            embedding_cache_stats = embedder.get_cache_stats()
        
        return {
            "status": "success",
            "vector_store_documents": count,
            "bm25_indexed_documents": bm25_count,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "embedding_cache": embedding_cache_stats
        }
        
    except Exception as e:
        logger.error("Error getting database info", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=True
    )
