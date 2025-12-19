"""Optional cross-encoder reranker for retrieval results.

This improves grounding by reranking candidate chunks with a query-document
cross-encoder model (e.g. BGE reranker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

from backend.config import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    item: Dict[str, Any]
    rerank_score: float


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.
    Designed to be optional: if model can't load, caller can disable reranking.
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or config.RERANKER_MODEL
        self.device = device
        self._model = None

        try:
            # sentence-transformers is already a dependency (used for embeddings)
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Loaded reranker model: %s", self.model_name)
        except Exception as e:
            logger.warning("Reranker disabled (failed to load %s): %s", self.model_name, e)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        items: Sequence[Dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank items by cross-encoder score (descending).

        Returns the same dicts with an added 'rerank_score' field.
        """
        if not items:
            return []
        if not self.available:
            return list(items[:top_k]) if top_k is not None else list(items)

        pairs: List[Tuple[str, str]] = []
        for it in items:
            pairs.append((query, str(it.get(text_key, "") or "")))

        try:
            scores = self._model.predict(pairs)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Reranker predict failed; returning un-reranked items: %s", e)
            return list(items[:top_k]) if top_k is not None else list(items)

        enriched: List[Dict[str, Any]] = []
        for it, sc in zip(items, scores):
            d = dict(it)
            d["rerank_score"] = float(sc)
            enriched.append(d)

        enriched.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        if top_k is not None:
            enriched = enriched[:top_k]
        return enriched


