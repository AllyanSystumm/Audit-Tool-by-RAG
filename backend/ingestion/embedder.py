"""Embedding generation using sentence transformers."""

from typing import List, Union
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers."""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max = int(getattr(config, "EMBEDDING_CACHE_MAX", 2048))
        
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        try:
            out: List[np.ndarray] = []
            missing_texts: List[str] = []
            missing_idxs: List[int] = []

            for i, t in enumerate(texts):
                key = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()
                emb = self._cache.get(key)
                if emb is None:
                    missing_texts.append(t)
                    missing_idxs.append(i)
                    out.append(None)  
                else:
                    out.append(emb)

            if missing_texts:
                new_embs = self.model.encode(
                    missing_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                for t, idx, e in zip(missing_texts, missing_idxs, new_embs):
                    key = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()
                    e = np.asarray(e)
                    self._cache[key] = e
                    out[idx] = e

                if len(self._cache) > self._cache_max:
                    for k in list(self._cache.keys())[: max(0, len(self._cache) - self._cache_max)]:
                        self._cache.pop(k, None)

            embeddings = np.vstack([np.asarray(e) for e in out]) if out else np.array([])
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def encode_chunks(
        self,
        chunks: List[dict],
        text_key: str = 'text',
        batch_size: int = 32
    ) -> List[dict]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key in chunk dict containing text
            batch_size: Batch size for encoding
            
        Returns:
            Chunks with added 'embedding' key
        """
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
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """

        return float(np.dot(embedding1, embedding2))
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = np.dot(candidate_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self):
        """Initialize embedding cache."""
        self.cache = {}
    
    def get(self, text: str) -> Union[np.ndarray, None]:
        """Get embedding from cache."""
        key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        return self.cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        self.cache[key] = embedding
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


if __name__ == "__main__":
    embedder = EmbeddingGenerator()

    texts = [
        "Quality management is essential for success.",
        "Verification processes ensure compliance.",
        "Documentation must meet requirements."
    ]

    embeddings = embedder.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.3f}")

