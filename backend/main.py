"""FastAPI backend for Audit Tool RAG application."""

import os
import sys
from pathlib import Path
from typing import Optional
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException
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
from backend.generation.llm_client import LLMClient
from backend.generation.checkpoint_generator import CheckpointGenerator
from backend.generation.process_profiles import detect_process_type
from backend.generation.prompts import _extract_headings_and_excerpts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy Chroma telemetry logger (does not affect functionality)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# If KEEP_UPLOADED_FILES is enabled, keep copies of uploaded files for review.
def _cleanup_uploaded_file(path: Path):
    try:
        if config.KEEP_UPLOADED_FILES:
            logger.info(f"KEEP_UPLOADED_FILES=true; keeping uploaded file at: {path}")
            return
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        logger.exception(f"Failed to clean up uploaded file: {path}")

# -----------------------------
# Helpers (KB / Retrieval)
# -----------------------------

def _rebuild_bm25_from_vector_store() -> int:
    """
    Rebuild BM25 index from all documents in the persistent Chroma collection.
    This keeps BM25 in sync across restarts and across multiple ingested documents.
    """
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
    """
    Merge multiple hybrid result lists by id, keeping max hybrid_score per id.
    """
    merged = {}
    for results in results_lists:
        for r in results or []:
            rid = r.get("id") or r.get("text", "")[:100]
            if not rid:
                continue
            if rid not in merged or r.get("hybrid_score", 0) > merged[rid].get("hybrid_score", 0):
                merged[rid] = r
    ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)

    # Diversity heuristic: avoid returning too many chunks from the same source file
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

# Initialize FastAPI app
app = FastAPI(
    title="Audit Tool API",
    description="RAG-powered checkpoint generation for audit processes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized at startup)
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
    """Initialize components on startup."""
    global document_processor, chunker, embedder, vector_store
    global bm25_retriever, hybrid_retriever, llm_client, checkpoint_generator
    
    logger.info("Initializing Audit Tool components...")
    
    try:
        # Initialize components
        document_processor = DocumentProcessor()
        chunker = SemanticChunker()
        embedder = EmbeddingGenerator()
        vector_store = VectorStore()
        bm25_retriever = BM25Retriever()
        hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)

        # Rebuild BM25 index from persisted Chroma documents (so KB retrieval works after restart)
        existing_count = vector_store.get_count()
        if existing_count > 0:
            rebuilt = _rebuild_bm25_from_vector_store()
            logger.info(f"Rebuilt BM25 index with {rebuilt} documents from Chroma")
        
        # Initialize LLM (only if token is available)
        if config.HF_TOKEN:
            llm_client = LLMClient()
            checkpoint_generator = CheckpointGenerator(llm_client)
            logger.info("LLM client initialized successfully")
        else:
            logger.warning("HF_TOKEN not set. LLM features will be disabled.")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        # Continue anyway - endpoints will handle missing components


@app.get("/")
def read_root():
    """Root endpoint with API information."""
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "document_processor": document_processor is not None,
            "embedder": embedder is not None,
            "vector_store": vector_store is not None,
            "llm_client": llm_client is not None,
            "vector_db_count": vector_store.get_count() if vector_store else 0
        }
    }


@app.post("/ingest-documents")
async def ingest_documents(file: UploadFile = File(...)):
    """
    Ingest a document into the vector database.
    This endpoint is for building the knowledge base.
    """
    if not all([document_processor, chunker, embedder, vector_store, bm25_retriever]):
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        # Save uploaded file
        file_path = config.UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Extract text
        extraction_result = document_processor.extract_text(str(file_path))
        
        if extraction_result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text: {extraction_result.get('error', 'Unknown error')}"
            )
        
        text = extraction_result['text']
        metadata = extraction_result['metadata']

        # Attach process_type metadata for retrieval filtering / KB hygiene
        ptype = detect_process_type(filename=file.filename, text=text)
        metadata = {**(metadata or {}), "process_type": ptype}
        
        # Chunk text
        chunks = chunker.chunk_text(text, metadata)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunks_with_embeddings = embedder.encode_chunks(chunks)
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        
        # Add to vector store (stable ids → dedupe + upsert on re-ingest)
        doc_id = hashlib.sha1(content).hexdigest()[:12]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, embeddings, ids=ids)
        
        # Rebuild BM25 so it includes the whole knowledge base (not just this file)
        bm25_count = _rebuild_bm25_from_vector_store()
        logger.info(f"BM25 index now has {bm25_count} documents")
        
        # Clean up uploaded file (or keep it if configured)
        _cleanup_uploaded_file(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_documents_in_db": vector_store.get_count(),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-checkpoints")
async def generate_checkpoints(
    file: UploadFile = File(...),
    use_rag: bool = True,
    num_checkpoints: int = 5,
    ingest_to_kb: bool = False,
    process_type: str = "auto"
):
    """
    Generate verification checkpoints for a process document.
    
    Args:
        file: Process document file
        use_rag: Whether to use RAG (retrieve similar context)
        num_checkpoints: Number of checkpoints to generate
    """
    if not checkpoint_generator:
        raise HTTPException(
            status_code=500,
            detail="LLM client not initialized. Please set HF_TOKEN environment variable."
        )
    
    if not all([document_processor, chunker, embedder, vector_store, bm25_retriever, hybrid_retriever]):
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        # Save uploaded file
        file_path = config.UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Generating checkpoints for: {file.filename}")
        
        # Extract text from uploaded document
        extraction_result = document_processor.extract_text(str(file_path))
        
        if extraction_result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text: {extraction_result.get('error', 'Unknown error')}"
            )
        
        process_document = extraction_result['text']
        metadata = extraction_result['metadata']

        # Resolve process_type for retrieval filtering / ingestion metadata
        resolved_ptype = process_type if process_type and process_type != "auto" else detect_process_type(
            filename=file.filename,
            text=process_document
        )
        filter_metadata = {"process_type": resolved_ptype} if resolved_ptype and resolved_ptype != "auto" else None
        
        # Optionally ingest the uploaded document into the Knowledge Base (useful for future retrieval)
        if ingest_to_kb:
            logger.info("Ingesting uploaded document into Knowledge Base...")
            ingest_md = {**(metadata or {}), "process_type": resolved_ptype}
            chunks = chunker.chunk_text(process_document, ingest_md)
            logger.info(f"Created {len(chunks)} chunks")

            texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(texts)

            # Stable doc id so repeated runs don't collide (used for upsert)
            doc_id = hashlib.sha1(content).hexdigest()[:12]
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

            vector_store.add_documents(chunks, embeddings, ids=ids)
            logger.info(f"Stored {len(chunks)} chunks in vector database")

            bm25_count = _rebuild_bm25_from_vector_store()
            logger.info(f"BM25 index now has {bm25_count} documents")
        
        # Retrieve relevant context if RAG is enabled
        relevant_chunks = []
        retrieved_chunks_preview = []
        if use_rag and vector_store.get_count() > 0:
            try:
                logger.info("Retrieving relevant context using hybrid search...")
                # Build multiple queries from detected headings/excerpts + a global snippet.
                heading_pairs = _extract_headings_and_excerpts(process_document, max_items=6)
                queries = []
                for h, ex in heading_pairs[:5]:
                    queries.append(f"{h}\n{ex}")
                # Always include a short global query too
                queries.append(process_document[:1800])
                # Fallback: ensure at least one query exists
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
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

                # Build a small preview for traceability/debugging (returned to UI)
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
        
        # Generate checkpoints
        result = checkpoint_generator.generate_checkpoints(
            process_document,
            relevant_chunks,
            num_checkpoints,
            process_type=process_type,  # "auto" | "verification" | "joint_review" | "configuration_management"
            filename=file.filename
        )
        
        # Clean up uploaded file (or keep it if configured)
        _cleanup_uploaded_file(file_path)
        
        if result['status'] != 'success':
            raise HTTPException(
                status_code=500,
                detail=f"Error generating checkpoints: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "status": "success",
            "filename": file.filename,
            "checkpoints": result['checkpoints'],
            "num_checkpoints": result['num_checkpoints'],
            "used_rag": use_rag and len(relevant_chunks) > 0,
            "num_context_chunks": len(relevant_chunks),
            "ingest_to_kb": ingest_to_kb,
            "retrieved_chunks": retrieved_chunks_preview,
            "process_type": result.get("process_type", "auto"),
            "process_profile": result.get("process_profile"),
            "process_doc_score": result.get("process_doc_score"),
            "raw_output": result.get('raw_output', '')  # Include raw output for debugging
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
    """
    Simple upload endpoint for backward compatibility.
    Extracts text from document.
    """
    if not document_processor:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    try:
        # Save uploaded file
        file_path = config.UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text
        result = document_processor.extract_text(str(file_path))
        
        # Clean up uploaded file (or keep it if configured)
        _cleanup_uploaded_file(file_path)
        
        if result['status'] != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Error: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "status": "success",
            "filename": file.filename,
            "text": result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
            "full_text_length": len(result['text']),
            "metadata": result['metadata']
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-database")
async def reset_database():
    """Reset the vector database (use with caution!)."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        vector_store.reset_collection()
        
        # Reinitialize BM25
        global bm25_retriever
        bm25_retriever = BM25Retriever()
        # Ensure hybrid retriever uses the new BM25 instance
        global hybrid_retriever
        hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
        
        return {
            "status": "success",
            "message": "Database reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/database-info")
def get_database_info():
    """Get information about the vector database."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        count = vector_store.get_count()
        bm25_count = bm25_retriever.get_document_count() if bm25_retriever else 0
        
        return {
            "status": "success",
            "vector_store_documents": count,
            "bm25_indexed_documents": bm25_count,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL
        }
        
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=True
    )
