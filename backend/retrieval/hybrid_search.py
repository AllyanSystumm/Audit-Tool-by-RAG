"""Hybrid search combining vector similarity and BM25."""

from typing import List, Dict, Optional
import numpy as np
import logging

from backend.config import config
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity search and BM25.
    
    Uses a weighted combination of both methods for better retrieval quality.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        alpha: float = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore instance
            bm25_retriever: BM25Retriever instance
            alpha: Weight for vector search (0-1). BM25 weight = 1-alpha
                   Default: 0.5 (equal weighting)
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha if alpha is not None else config.HYBRID_SEARCH_ALPHA
        # Optional cross-encoder reranker (best-effort: may be unavailable if model can't load)
        self.reranker = reranker if reranker is not None else (
            CrossEncoderReranker() if getattr(config, "ENABLE_RERANKER", False) else None
        )
        
        logger.info(f"Initialized hybrid retriever with alpha={self.alpha}")
    
    def search(
        self,
        query: str,
        embedder,
        top_k: int = None,
        alpha: float = None,
        filter_metadata: Optional[Dict] = None,
        use_reranker: bool = True
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25.
        
        Args:
            query: Search query
            embedder: EmbeddingGenerator instance
            top_k: Number of results to return
            alpha: Override alpha for this search
            
        Returns:
            List of ranked results
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        alpha = alpha if alpha is not None else self.alpha
        
        try:
            # Get more results from each method to ensure good coverage after merging/reranking
            retrieval_k = min(max(top_k * 4, int(getattr(config, "RERANK_TOP_N", 50))), 80)
            # Avoid asking Chroma for more results than exist (reduces warnings/noise and speeds small KBs)
            try:
                count = int(self.vector_store.get_count())
                if count > 0:
                    retrieval_k = min(retrieval_k, count)
            except Exception:
                pass
            
            # Vector search
            vector_results = self.vector_store.search_by_text(
                query,
                embedder,
                top_k=retrieval_k,
                filter_metadata=filter_metadata
            )
            
            # BM25 search
            bm25_results = self.bm25_retriever.search(
                query,
                top_k=retrieval_k,
                filter_metadata=filter_metadata
            )
            
            # Combine and rerank
            hybrid_results = self._combine_results(
                vector_results,
                bm25_results,
                alpha=alpha
            )

            # Optional cross-encoder reranking (true semantic reranking)
            if use_reranker and self.reranker is not None and getattr(self.reranker, "available", False):
                rerank_n = min(int(getattr(config, "RERANK_TOP_N", 50)), len(hybrid_results))
                candidates = hybrid_results[:rerank_n]
                reranked = self.reranker.rerank(query, candidates, text_key="text", top_k=rerank_n)
                # Stable ordering: rerank_score primary, hybrid_score secondary
                reranked.sort(
                    key=lambda x: (x.get("rerank_score", float("-inf")), x.get("hybrid_score", 0.0)),
                    reverse=True
                )
                # Append non-reranked tail (already sorted by hybrid_score)
                hybrid_results = reranked + hybrid_results[rerank_n:]
            
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def _combine_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float
    ) -> List[Dict]:
        """
        Combine and rerank results from both methods.
        
        Uses Reciprocal Rank Fusion (RRF) combined with score-based fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for vector scores
            
        Returns:
            Combined and reranked results
        """
        # Normalize scores to [0, 1] range
        vector_scores = self._normalize_scores(
            [r.get('score', 1 - r.get('distance', 0)) for r in vector_results]
        )
        bm25_scores = self._normalize_scores(
            [r.get('bm25_score', 0) for r in bm25_results]
        )
        
        # Create document map with combined scores
        doc_map = {}
        
        # Add vector results
        for i, result in enumerate(vector_results):
            doc_id = result.get('id') or result.get('text')[:100]
            doc_map[doc_id] = {
                **result,
                'vector_score': vector_scores[i],
                'bm25_score': 0,
                'vector_rank': i + 1
            }
        
        # Add/update with BM25 results
        for i, result in enumerate(bm25_results):
            doc_id = result.get('id') or result.get('text')[:100]
            
            if doc_id in doc_map:
                # Document found in both - update BM25 score
                doc_map[doc_id]['bm25_score'] = bm25_scores[i]
                doc_map[doc_id]['bm25_rank'] = i + 1
            else:
                # New document from BM25
                doc_map[doc_id] = {
                    **result,
                    'vector_score': 0,
                    'bm25_score': bm25_scores[i],
                    'bm25_rank': i + 1
                }
        
        # Compute hybrid scores
        for doc_id, doc in doc_map.items():
            # Score-based fusion
            score_fusion = (
                alpha * doc.get('vector_score', 0) +
                (1 - alpha) * doc.get('bm25_score', 0)
            )
            
            # Reciprocal Rank Fusion (RRF)
            # RRF(d) = sum(1 / (k + rank(d))) where k=60 is standard
            k = 60
            vector_rank = doc.get('vector_rank', 1000)
            bm25_rank = doc.get('bm25_rank', 1000)
            rrf_score = (1 / (k + vector_rank)) + (1 / (k + bm25_rank))
            
            # Combine both methods (weighted average)
            doc['hybrid_score'] = 0.7 * score_fusion + 0.3 * (rrf_score * 100)
        
        # Sort by hybrid score
        ranked_results = sorted(
            doc_map.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        logger.info(f"Combined {len(vector_results)} vector + {len(bm25_results)} BM25 results into {len(ranked_results)} hybrid results")
        
        return ranked_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()


# Example usage
if __name__ == "__main__":
    from ingestion.embedder import EmbeddingGenerator
    
    # Initialize components
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    embedder = EmbeddingGenerator()
    
    # Sample documents
    documents = [
        {'text': 'Quality management ensures compliance.', 'metadata': {}},
        {'text': 'Verification is a critical process.', 'metadata': {}},
        {'text': 'ISO 9001 standards for quality.', 'metadata': {}}
    ]
    
    # Index documents
    embeddings = embedder.encode([d['text'] for d in documents])
    vector_store.add_documents(documents, embeddings)
    bm25_retriever.index_documents(documents)
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
    
    # Search
    query = "quality verification standards"
    results = hybrid_retriever.search(query, embedder, top_k=3)
    
    print(f"Top results for '{query}':")
    for result in results:
        print(f"- {result['text']} (score: {result['hybrid_score']:.3f})")

