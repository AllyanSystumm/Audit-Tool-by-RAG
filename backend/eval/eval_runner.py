"""Lightweight evaluation harness for checkpoint generation quality.

Usage:
  python -m backend.eval.eval_runner

This script:
- resets the vector DB
- ingests all documents under data/uploads/
- runs generation for each uploaded document with RAG enabled
- prints basic metrics (schema validity, profile adherence, canonical prompt anchoring)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from backend.config import config
from backend.ingestion.document_processor import DocumentProcessor
from backend.ingestion.chunker import SemanticChunker
from backend.ingestion.embedder import EmbeddingGenerator
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_search import HybridRetriever
from backend.generation.llm_client import LLMClient
from backend.generation.checkpoint_generator import CheckpointGenerator
from backend.generation.process_profiles import get_profile, detect_process_type

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".xls", ".xml", ".pptx", ".txt"}


def _is_supported_file(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _ingest_file(path: Path, processor: DocumentProcessor, chunker: SemanticChunker, embedder: EmbeddingGenerator, store: VectorStore) -> int:
    res = processor.extract_text(str(path))
    if res.get("status") != "success":
        raise RuntimeError(f"Failed to extract {path.name}: {res.get('error')}")
    text = res["text"]
    md = res.get("metadata") or {}
    md["process_type"] = detect_process_type(filename=path.name, text=text)

    chunks = chunker.chunk_text(text, md)
    embeddings = embedder.encode([c["text"] for c in chunks])
    content = path.read_bytes()
    doc_id = hashlib.sha1(content).hexdigest()[:12]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    store.add_documents(chunks, embeddings, ids=ids)
    return len(chunks)


def main():
    uploads = (config.DATA_DIR / "uploads")
    all_files = sorted([p for p in uploads.iterdir() if p.is_file()])
    files = [f for f in all_files if _is_supported_file(f)]
    
    skipped = len(all_files) - len(files)
    if skipped > 0:
        print(f"Skipping {skipped} unsupported file(s)")
    
    if not files:
        print(f"No supported files found under {uploads}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    processor = DocumentProcessor()
    chunker = SemanticChunker()
    embedder = EmbeddingGenerator()
    store = VectorStore()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(store, bm25)

    store.reset_collection()

    total_chunks = 0
    for f in files:
        try:
            n = _ingest_file(f, processor, chunker, embedder, store)
            total_chunks += n
            print(f"Ingested {f.name}: {n} chunks")
        except Exception as e:
            print(f"Failed to ingest {f.name}: {e}")

    all_docs = store.get_all_documents()
    docs = all_docs.get("documents") or []
    metas = all_docs.get("metadatas") or []
    ids = all_docs.get("ids") or []
    bm25_docs = []
    for i, t in enumerate(docs):
        bm25_docs.append({"text": t, "metadata": metas[i] if i < len(metas) else {}, "id": ids[i] if i < len(ids) else None})
    bm25.index_documents(bm25_docs)

    print(f"\nKB ready: vector_store={store.get_count()} chunks (ingested {total_chunks})\n")

    if not config.HF_TOKEN:
        print("HF_TOKEN is not set; cannot run LLM generation eval.")
        return

    llm = LLMClient()
    gen = CheckpointGenerator(llm)

    for f in files:
        res = processor.extract_text(str(f))
        if res.get("status") != "success":
            print(f"Skipping {f.name}: extraction failed")
            continue
            
        text = res["text"]
        resolved_ptype = detect_process_type(filename=f.name, text=text)
        profile = get_profile(process_type="auto", filename=f.name, text=text)
        filter_md = {"process_type": resolved_ptype} if resolved_ptype and resolved_ptype != "auto" else None

        retrieved = hybrid.search(text[:1600], embedder, top_k=config.TOP_K_RETRIEVAL, filter_metadata=filter_md, use_reranker=True)

        out = gen.generate_checkpoints(
            process_document=text,
            relevant_chunks=retrieved,
            num_checkpoints=5,
            process_type="auto",
            filename=f.name
        )

        cps = out.get("checkpoints") or []
        print("=" * 90)
        print(f"{f.name} | detected={resolved_ptype} | profile={out.get('process_profile')} | cps={len(cps)} | used_rag={len(retrieved) > 0}")

        schema_ok = all(
            isinstance(cp, dict)
            and all(k in cp for k in ("process_phase_reference", "standard_clause_reference", "verification_section", "prompt"))
            for cp in cps
        )
        print(f"- schema_ok: {schema_ok}")

        if profile is not None and cps:
            slot_by_phase = {s.process_phase_reference: s for s in profile.slots}
            anchored = 0
            for cp in cps:
                phase = cp.get("process_phase_reference", "")
                slot = slot_by_phase.get(phase)
                if not slot:
                    continue
                if str(cp.get("prompt", "")).strip().lower().startswith(slot.canonical_prompt.strip().lower()):
                    anchored += 1
            print(f"- canonical_prompt_anchored: {anchored}/{len(cps)}")


if __name__ == "__main__":
    main()
