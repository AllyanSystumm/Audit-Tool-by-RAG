"""Hybrid search combining vector similarity and BM25."""

from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import logging

from backend.config import config
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearchTimeoutError(Exception):
    pass


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        alpha: float = None,
        reranker: Optional[CrossEncoderReranker] = None,
        search_timeout: float = None
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha if alpha is not None else config.HYBRID_SEARCH_ALPHA
        self.reranker = reranker if reranker is not None else (
            CrossEncoderReranker() if getattr(config, "ENABLE_RERANKER", False) else None
        )
        self.search_timeout = search_timeout or getattr(config, "EXTERNAL_REQUEST_TIMEOUT", 30.0)
        
        logger.info("Initialized hybrid retriever with alpha=%.2f, timeout=%.1fs", self.alpha, self.search_timeout)

    def _vector_search_with_timeout(
        self,
        query: str,
        embedder,
        top_k: int,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        return self.vector_store.search_by_text(
            query,
            embedder,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

    def _bm25_search_with_timeout(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        return self.bm25_retriever.search(
            query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

    def search(
        self,
        query: str,
        embedder,
        top_k: int = None,
        alpha: float = None,
        filter_metadata: Optional[Dict] = None,
        use_reranker: bool = True,
        timeout: float = None
    ) -> List[Dict]:
        top_k = top_k or config.TOP_K_RETRIEVAL
        alpha = alpha if alpha is not None else self.alpha
        timeout = timeout or self.search_timeout
        
        try:
            retrieval_k = min(max(top_k * 4, int(getattr(config, "RERANK_TOP_N", 50))), 80)

            try:
                count = int(self.vector_store.get_count())
                if count > 0:
                    retrieval_k = min(retrieval_k, count)
            except Exception:
                pass

            vector_results = []
            bm25_results = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(
                    self._vector_search_with_timeout,
                    query, embedder, retrieval_k, filter_metadata
                )
                bm25_future = executor.submit(
                    self._bm25_search_with_timeout,
                    query, retrieval_k, filter_metadata
                )

                try:
                    vector_results = vector_future.result(timeout=timeout)
                except FuturesTimeoutError:
                    logger.warning("Vector search timed out after %.1fs", timeout)
                    vector_results = []
                except Exception as e:
                    logger.warning("Vector search failed: %s", str(e))
                    vector_results = []

                try:
                    bm25_results = bm25_future.result(timeout=timeout)
                except FuturesTimeoutError:
                    logger.warning("BM25 search timed out after %.1fs", timeout)
                    bm25_results = []
                except Exception as e:
                    logger.warning("BM25 search failed: %s", str(e))
                    bm25_results = []

            if not vector_results and not bm25_results:
                logger.warning("Both search methods returned no results")
                return []

            hybrid_results = self._combine_results(
                vector_results,
                bm25_results,
                alpha=alpha
            )

            if use_reranker and self.reranker is not None and getattr(self.reranker, "available", False):
                try:
                    rerank_n = min(int(getattr(config, "RERANK_TOP_N", 50)), len(hybrid_results))
                    candidates = hybrid_results[:rerank_n]
                    reranked = self.reranker.rerank(query, candidates, text_key="text", top_k=rerank_n)
                    reranked.sort(
                        key=lambda x: (x.get("rerank_score", float("-inf")), x.get("hybrid_score", 0.0)),
                        reverse=True
                    )
                    hybrid_results = reranked + hybrid_results[rerank_n:]
                except Exception as e:
                    logger.warning("Reranking failed, using hybrid scores: %s", str(e))
            
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error("Error in hybrid search: %s", str(e))
            raise
    
    def _combine_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float
    ) -> List[Dict]:
        vector_scores = self._normalize_scores(
            [r.get('score', 1 - r.get('distance', 0)) for r in vector_results]
        )
        bm25_scores = self._normalize_scores(
            [r.get('bm25_score', 0) for r in bm25_results]
        )

        doc_map = {}

        for i, result in enumerate(vector_results):
            doc_id = result.get('id') or result.get('text')[:100]
            doc_map[doc_id] = {
                **result,
                'vector_score': vector_scores[i],
                'bm25_score': 0,
                'vector_rank': i + 1
            }

        for i, result in enumerate(bm25_results):
            doc_id = result.get('id') or result.get('text')[:100]
            
            if doc_id in doc_map:
                doc_map[doc_id]['bm25_score'] = bm25_scores[i]
                doc_map[doc_id]['bm25_rank'] = i + 1
            else:
                doc_map[doc_id] = {
                    **result,
                    'vector_score': 0,
                    'bm25_score': bm25_scores[i],
                    'bm25_rank': i + 1
                }

        for doc_id, doc in doc_map.items():
            score_fusion = (
                alpha * doc.get('vector_score', 0) +
                (1 - alpha) * doc.get('bm25_score', 0)
            )

            k = 60
            vector_rank = doc.get('vector_rank', 1000)
            bm25_rank = doc.get('bm25_rank', 1000)
            rrf_score = (1 / (k + vector_rank)) + (1 / (k + bm25_rank))

            doc['hybrid_score'] = 0.7 * score_fusion + 0.3 * (rrf_score * 100)

        ranked_results = sorted(
            doc_map.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        logger.info("Combined %d vector + %d BM25 results into %d hybrid results", 
                   len(vector_results), len(bm25_results), len(ranked_results))
        
        return ranked_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()


if __name__ == "__main__":
    from ingestion.embedder import EmbeddingGenerator

    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    embedder = EmbeddingGenerator()

    documents = [
        {'text': 'Quality management ensures compliance.', 'metadata': {}},
        {'text': 'Verification is a critical process.', 'metadata': {}},
        {'text': 'ISO 9001 standards for quality.', 'metadata': {}}
    ]

    embeddings = embedder.encode([d['text'] for d in documents])
    vector_store.add_documents(documents, embeddings)
    bm25_retriever.index_documents(documents)

    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)

    query = "quality verification standards"
    results = hybrid_retriever.search(query, embedder, top_k=3)
    
    print(f"Top results for '{query}':")
    for result in results:
        print(f"- {result['text']} (score: {result['hybrid_score']:.3f})")
