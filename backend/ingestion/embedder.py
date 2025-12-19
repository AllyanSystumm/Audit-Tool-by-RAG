"""Embedding generation using sentence transformers."""

from typing import List, Union, OrderedDict
from collections import OrderedDict as OrderedDictType
import hashlib
import threading
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: OrderedDictType[str, np.ndarray] = OrderedDictType()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Union[np.ndarray, None]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2)
            }


class EmbeddingGenerator:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device
        self._cache = LRUCache(max_size=int(getattr(config, "EMBEDDING_CACHE_MAX", 2048)))
        
        logger.info("Loading embedding model: %s", self.model_name)
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info("Model loaded successfully. Embedding dimension: %d", self.embedding_dim)
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def _compute_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        try:
            out: List[np.ndarray] = [None] * len(texts)
            missing_texts: List[str] = []
            missing_idxs: List[int] = []

            for i, t in enumerate(texts):
                key = self._compute_cache_key(t)
                emb = self._cache.get(key)
                if emb is not None:
                    out[i] = emb
                else:
                    missing_texts.append(t)
                    missing_idxs.append(i)

            if missing_texts:
                new_embs = self.model.encode(
                    missing_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                for t, idx, e in zip(missing_texts, missing_idxs, new_embs):
                    key = self._compute_cache_key(t)
                    e = np.asarray(e)
                    self._cache.put(key, e)
                    out[idx] = e

            embeddings = np.vstack([np.asarray(e) for e in out]) if out else np.array([])
            logger.info("Generated %d embeddings (cache: %s)", len(embeddings), self._cache.stats())
            return embeddings
            
        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e))
            raise
    
    def encode_chunks(
        self,
        chunks: List[dict],
        text_key: str = 'text',
        batch_size: int = 32
    ) -> List[dict]:
        if not chunks:
            return []

        texts = [chunk[text_key] for chunk in chunks]
        embeddings = self.encode(texts, batch_size=batch_size)

        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        return float(np.dot(embedding1, embedding2))
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        similarities = np.dot(candidate_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results

    def get_cache_stats(self) -> dict:
        return self._cache.stats()

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Embedding cache cleared")


class EmbeddingCache:
    def __init__(self, max_size: int = 2048):
        self._lru = LRUCache(max_size=max_size)
    
    def get(self, text: str) -> Union[np.ndarray, None]:
        key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return self._lru.get(key)
    
    def set(self, text: str, embedding: np.ndarray):
        key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        self._lru.put(key, embedding)
    
    def clear(self):
        self._lru.clear()
    
    def size(self) -> int:
        return self._lru.size()

    def stats(self) -> dict:
        return self._lru.stats()


if __name__ == "__main__":
    embedder = EmbeddingGenerator()

    texts = [
        "Quality management is essential for success.",
        "Verification processes ensure compliance.",
        "Documentation must meet requirements."
    ]

    embeddings = embedder.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

    embeddings_cached = embedder.encode(texts)
    print(f"Cache stats: {embedder.get_cache_stats()}")

    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.3f}")
