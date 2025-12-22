"""LLM Benchmark module for comparing checkpoint generation quality across models.

Usage:
  python -m backend.eval.llm_benchmark

Generates a markdown report comparing LLM performance.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from backend.config import config
from backend.ingestion.document_processor import DocumentProcessor
from backend.ingestion.chunker import SemanticChunker
from backend.ingestion.embedder import EmbeddingGenerator
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_search import HybridRetriever
from backend.generation.llm_client import LLMClient, LLMClientFactory
from backend.generation.checkpoint_generator import CheckpointGenerator
from backend.generation.process_profiles import get_profile, detect_process_type

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".xls", ".xml", ".pptx", ".txt"}


@dataclass
class CheckpointMetrics:
    total_checkpoints: int = 0
    schema_valid: int = 0
    has_process_phase: int = 0
    has_standard_clause: int = 0
    has_verification_section: int = 0
    has_prompt: int = 0
    canonical_prompt_anchored: int = 0
    avg_prompt_length: float = 0.0


@dataclass
class DocumentResult:
    filename: str
    process_type: str
    process_profile: Optional[str]
    num_checkpoints: int
    metrics: CheckpointMetrics
    latency_seconds: float
    raw_output_length: int
    error: Optional[str] = None


@dataclass
class ModelBenchmarkResult:
    model_key: str
    model_id: str
    model_name: str
    document_results: List[DocumentResult] = field(default_factory=list)
    total_latency_seconds: float = 0.0
    avg_latency_seconds: float = 0.0
    total_checkpoints: int = 0
    schema_validity_rate: float = 0.0
    canonical_anchor_rate: float = 0.0
    avg_prompt_length: float = 0.0
    error_count: int = 0


@dataclass
class BenchmarkReport:
    timestamp: str
    models_tested: List[str]
    documents_tested: int
    results: Dict[str, ModelBenchmarkResult] = field(default_factory=dict)


def _is_supported_file(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _analyze_checkpoints(checkpoints: List[Dict], profile) -> CheckpointMetrics:
    metrics = CheckpointMetrics()
    metrics.total_checkpoints = len(checkpoints)
    
    if not checkpoints:
        return metrics
    
    prompt_lengths = []
    slot_by_phase = {}
    if profile is not None:
        slot_by_phase = {s.process_phase_reference: s for s in profile.slots}
    
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
            
        has_all_fields = all(
            k in cp for k in ("process_phase_reference", "standard_clause_reference", "verification_section", "prompt")
        )
        if has_all_fields:
            metrics.schema_valid += 1
        
        if cp.get("process_phase_reference"):
            metrics.has_process_phase += 1
        if cp.get("standard_clause_reference"):
            metrics.has_standard_clause += 1
        if cp.get("verification_section"):
            metrics.has_verification_section += 1
        
        prompt = cp.get("prompt", "")
        if prompt:
            metrics.has_prompt += 1
            prompt_lengths.append(len(prompt))
        
        phase = cp.get("process_phase_reference", "")
        slot = slot_by_phase.get(phase)
        if slot:
            canonical = slot.canonical_prompt.strip().lower()
            actual = str(prompt).strip().lower()
            if actual.startswith(canonical[:50]):
                metrics.canonical_prompt_anchored += 1
    
    if prompt_lengths:
        metrics.avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    
    return metrics


def _ingest_files(
    files: List[Path],
    processor: DocumentProcessor,
    chunker: SemanticChunker,
    embedder: EmbeddingGenerator,
    store: VectorStore,
    bm25: BM25Retriever
) -> int:
    store.reset_collection()
    total_chunks = 0
    
    for f in files:
        try:
            res = processor.extract_text(str(f))
            if res.get("status") != "success":
                continue
            text = res["text"]
            md = res.get("metadata") or {}
            md["process_type"] = detect_process_type(filename=f.name, text=text)
            
            chunks = chunker.chunk_text(text, md)
            embeddings = embedder.encode([c["text"] for c in chunks])
            content = f.read_bytes()
            doc_id = hashlib.sha1(content).hexdigest()[:12]
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            store.add_documents(chunks, embeddings, ids=ids)
            total_chunks += len(chunks)
        except Exception:
            continue
    
    all_docs = store.get_all_documents()
    docs = all_docs.get("documents") or []
    metas = all_docs.get("metadatas") or []
    ids_list = all_docs.get("ids") or []
    bm25_docs = []
    for i, t in enumerate(docs):
        bm25_docs.append({
            "text": t,
            "metadata": metas[i] if i < len(metas) else {},
            "id": ids_list[i] if i < len(ids_list) else None
        })
    bm25.index_documents(bm25_docs)
    
    return total_chunks


def run_benchmark(
    model_keys: List[str] = None,
    max_documents: int = None,
    verbose: bool = True
) -> BenchmarkReport:
    if model_keys is None:
        model_keys = list(config.LLM_MODELS.keys())
    
    uploads = config.DATA_DIR / "uploads"
    all_files = sorted([p for p in uploads.iterdir() if p.is_file()])
    files = [f for f in all_files if _is_supported_file(f)]
    
    if max_documents:
        files = files[:max_documents]
    
    if not files:
        raise ValueError(f"No supported files found under {uploads}")
    
    if verbose:
        print(f"Found {len(files)} document(s) for benchmark")
        print(f"Models to test: {model_keys}")
    
    processor = DocumentProcessor()
    chunker = SemanticChunker()
    embedder = EmbeddingGenerator()
    store = VectorStore()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(store, bm25)
    
    total_chunks = _ingest_files(files, processor, chunker, embedder, store, bm25)
    
    if verbose:
        print(f"Knowledge base ready: {total_chunks} chunks indexed")
    
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        models_tested=model_keys,
        documents_tested=len(files)
    )
    
    for model_key in model_keys:
        model_info = config.LLM_MODELS.get(model_key, {})
        model_id = config.get_llm_model_id(model_key)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_key} ({model_id})")
            print('='*60)
        
        result = ModelBenchmarkResult(
            model_key=model_key,
            model_id=model_id,
            model_name=model_info.get("name", model_key)
        )
        
        try:
            llm_client = LLMClientFactory.get_client(model_key)
            checkpoint_gen = CheckpointGenerator(llm_client)
        except Exception as e:
            if verbose:
                print(f"Failed to initialize model {model_key}: {e}")
            result.error_count = len(files)
            report.results[model_key] = result
            continue
        
        total_schema_valid = 0
        total_canonical_anchored = 0
        total_checkpoints = 0
        all_prompt_lengths = []
        
        for f in files:
            try:
                res = processor.extract_text(str(f))
                if res.get("status") != "success":
                    result.document_results.append(DocumentResult(
                        filename=f.name,
                        process_type="unknown",
                        process_profile=None,
                        num_checkpoints=0,
                        metrics=CheckpointMetrics(),
                        latency_seconds=0,
                        raw_output_length=0,
                        error="Extraction failed"
                    ))
                    result.error_count += 1
                    continue
                
                text = res["text"]
                resolved_ptype = detect_process_type(filename=f.name, text=text)
                profile = get_profile(process_type="auto", filename=f.name, text=text)
                filter_md = {"process_type": resolved_ptype} if resolved_ptype and resolved_ptype != "auto" else None
                
                retrieved = hybrid.search(
                    text[:1600],
                    embedder,
                    top_k=config.TOP_K_RETRIEVAL,
                    filter_metadata=filter_md,
                    use_reranker=True
                )
                
                start_time = time.time()
                out = checkpoint_gen.generate_checkpoints(
                    process_document=text,
                    relevant_chunks=retrieved,
                    num_checkpoints=5,
                    process_type="auto",
                    filename=f.name
                )
                latency = time.time() - start_time
                
                checkpoints = out.get("checkpoints") or []
                metrics = _analyze_checkpoints(checkpoints, profile)
                
                doc_result = DocumentResult(
                    filename=f.name,
                    process_type=resolved_ptype or "auto",
                    process_profile=out.get("process_profile"),
                    num_checkpoints=len(checkpoints),
                    metrics=metrics,
                    latency_seconds=latency,
                    raw_output_length=len(out.get("raw_output", ""))
                )
                result.document_results.append(doc_result)
                result.total_latency_seconds += latency
                
                total_checkpoints += metrics.total_checkpoints
                total_schema_valid += metrics.schema_valid
                total_canonical_anchored += metrics.canonical_prompt_anchored
                if metrics.avg_prompt_length > 0:
                    all_prompt_lengths.append(metrics.avg_prompt_length)
                
                if verbose:
                    print(f"  {f.name}: {len(checkpoints)} checkpoints, {latency:.2f}s, schema_valid={metrics.schema_valid}/{metrics.total_checkpoints}")
                
            except Exception as e:
                result.document_results.append(DocumentResult(
                    filename=f.name,
                    process_type="unknown",
                    process_profile=None,
                    num_checkpoints=0,
                    metrics=CheckpointMetrics(),
                    latency_seconds=0,
                    raw_output_length=0,
                    error=str(e)
                ))
                result.error_count += 1
                if verbose:
                    print(f"  {f.name}: ERROR - {e}")
        
        result.total_checkpoints = total_checkpoints
        if len(files) > 0:
            result.avg_latency_seconds = result.total_latency_seconds / len(files)
        if total_checkpoints > 0:
            result.schema_validity_rate = (total_schema_valid / total_checkpoints) * 100
            result.canonical_anchor_rate = (total_canonical_anchored / total_checkpoints) * 100
        if all_prompt_lengths:
            result.avg_prompt_length = sum(all_prompt_lengths) / len(all_prompt_lengths)
        
        report.results[model_key] = result
    
    return report


def generate_markdown_report(report: BenchmarkReport, output_path: Path = None) -> str:
    lines = []
    lines.append("# LLM Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Documents Tested:** {report.documents_tested}")
    lines.append(f"**Models Tested:** {', '.join(report.models_tested)}")
    lines.append("")
    
    lines.append("## Summary Comparison")
    lines.append("")
    lines.append("| Metric | " + " | ".join(report.models_tested) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(report.models_tested)) + "|")
    
    metrics_rows = [
        ("Total Checkpoints", lambda r: str(r.total_checkpoints)),
        ("Schema Validity Rate", lambda r: f"{r.schema_validity_rate:.1f}%"),
        ("Canonical Anchor Rate", lambda r: f"{r.canonical_anchor_rate:.1f}%"),
        ("Avg Latency (s)", lambda r: f"{r.avg_latency_seconds:.2f}"),
        ("Avg Prompt Length", lambda r: f"{r.avg_prompt_length:.0f}"),
        ("Error Count", lambda r: str(r.error_count)),
    ]
    
    for metric_name, getter in metrics_rows:
        values = []
        for model_key in report.models_tested:
            result = report.results.get(model_key)
            if result:
                values.append(getter(result))
            else:
                values.append("N/A")
        lines.append(f"| {metric_name} | " + " | ".join(values) + " |")
    
    lines.append("")
    lines.append("## Detailed Results by Model")
    lines.append("")
    
    for model_key in report.models_tested:
        result = report.results.get(model_key)
        if not result:
            continue
        
        lines.append(f"### {result.model_name}")
        lines.append("")
        lines.append(f"- **Model ID:** `{result.model_id}`")
        lines.append(f"- **Total Latency:** {result.total_latency_seconds:.2f}s")
        lines.append(f"- **Average Latency:** {result.avg_latency_seconds:.2f}s per document")
        lines.append("")
        
        lines.append("#### Document Results")
        lines.append("")
        lines.append("| Document | Process Type | Checkpoints | Schema Valid | Latency (s) | Status |")
        lines.append("|----------|--------------|-------------|--------------|-------------|--------|")
        
        for doc in result.document_results:
            status = "✅" if not doc.error else f"❌ {doc.error[:20]}..."
            lines.append(
                f"| {doc.filename} | {doc.process_type} | {doc.num_checkpoints} | "
                f"{doc.metrics.schema_valid}/{doc.metrics.total_checkpoints} | "
                f"{doc.latency_seconds:.2f} | {status} |"
            )
        
        lines.append("")
    
    lines.append("## Quality Metrics Explanation")
    lines.append("")
    lines.append("- **Schema Validity Rate:** Percentage of checkpoints with all required fields")
    lines.append("- **Canonical Anchor Rate:** Percentage of checkpoints starting with expected canonical prompts")
    lines.append("- **Avg Prompt Length:** Average character length of generated prompts")
    lines.append("")
    
    lines.append("## Recommendations")
    lines.append("")
    
    if len(report.results) >= 2:
        best_quality = max(
            report.results.values(),
            key=lambda r: r.schema_validity_rate + r.canonical_anchor_rate
        )
        best_speed = min(
            report.results.values(),
            key=lambda r: r.avg_latency_seconds if r.avg_latency_seconds > 0 else float('inf')
        )
        
        lines.append(f"- **Best Quality:** {best_quality.model_name} (Schema: {best_quality.schema_validity_rate:.1f}%, Anchor: {best_quality.canonical_anchor_rate:.1f}%)")
        lines.append(f"- **Fastest:** {best_speed.model_name} ({best_speed.avg_latency_seconds:.2f}s avg)")
        
        if best_quality.model_key == best_speed.model_key:
            lines.append(f"- **Recommendation:** Use **{best_quality.model_name}** - best in both quality and speed")
        else:
            lines.append(f"- **Recommendation:** Use **{best_quality.model_name}** for production (better quality)")
    
    lines.append("")
    
    content = "\n".join(lines)
    
    if output_path:
        output_path.write_text(content)
    
    return content


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN is not set; cannot run LLM benchmark.")
        return
    
    print("Starting LLM Benchmark...")
    print("")
    
    report = run_benchmark(verbose=True)
    
    output_path = config.BASE_DIR / "benchmark_report.md"
    markdown = generate_markdown_report(report, output_path)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nReport saved to: {output_path}")
    print("\nQuick Summary:")
    for model_key, result in report.results.items():
        print(f"  {result.model_name}:")
        print(f"    - Checkpoints: {result.total_checkpoints}")
        print(f"    - Schema Validity: {result.schema_validity_rate:.1f}%")
        print(f"    - Avg Latency: {result.avg_latency_seconds:.2f}s")


if __name__ == "__main__":
    main()

