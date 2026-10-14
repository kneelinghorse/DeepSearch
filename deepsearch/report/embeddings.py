"""Embedding-based semantic matching for citations.

This module provides semantic similarity matching using local embeddings
via Ollama, with graceful fallback to keyword matching when unavailable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = os.environ.get("DEEPSEARCH_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.environ.get("DEEPSEARCH_SIMILARITY_THRESHOLD", "0.7")
)
USE_SEMANTIC_MATCHING = os.environ.get(
    "DEEPSEARCH_USE_SEMANTIC_MATCHING", "true"
).lower() in ("true", "1", "yes")

# B35.3: when two semantic-similarity scores are within this epsilon, the
# source with the higher authority_score wins. Tight epsilon (~2% of cosine
# range) so authority only breaks meaningful ties — sources whose semantic
# match is materially worse still lose.
_AUTHORITY_TIE_EPSILON = 0.02


def _authority_for(source: Mapping[str, object]) -> float:
    """Best-effort authority_score read; defaults to 0.0 when unset."""

    raw = source.get("authority_score")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

# Lazy numpy import for environments without it
_np = None


def _get_numpy():
    """Lazy import numpy to avoid startup cost if not used."""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            logger.warning("numpy not installed, semantic matching unavailable")
            _np = False
    return _np if _np else None


@dataclass
class EmbeddingResult:
    """Result of embedding computation."""
    text: str
    vector: "np.ndarray"  # type: ignore
    model: str


class OllamaEmbedder:
    """Generate embeddings using local Ollama.

    Attributes:
        model: The Ollama model to use for embeddings
        base_url: URL of the Ollama API server
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._available: Optional[bool] = None
        self._cache: Dict[str, "np.ndarray"] = {}  # type: ignore

    @property
    def available(self) -> bool:
        """Check if Ollama is available and numpy is installed."""
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """Check if Ollama service is running and numpy available."""
        np = _get_numpy()
        if np is None:
            return False

        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Ollama not available: %s", e)
            return False

    def embed(self, text: str) -> Optional["np.ndarray"]:  # type: ignore
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding vector, or None if unavailable
        """
        if not self.available:
            return None

        # Check cache first
        cache_key = f"{self.model}:{text[:500]}"  # Truncate for cache key
        if cache_key in self._cache:
            return self._cache[cache_key]

        np = _get_numpy()
        if np is None:
            return None

        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            vector = np.array(resp.json()["embedding"])
            self._cache[cache_key] = vector
            return vector
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional["np.ndarray"]]:  # type: ignore
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        return [self.embed(t) for t in texts]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:  # type: ignore
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    np = _get_numpy()
    if np is None:
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticMatcher:
    """Match findings to sources using semantic similarity.

    This class provides semantic matching using embeddings, with automatic
    fallback to keyword matching when embeddings are unavailable.

    Attributes:
        embedder: OllamaEmbedder instance for generating embeddings
        threshold: Minimum similarity score for a match (0-1)
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.threshold = similarity_threshold
        self._source_embeddings: Dict[int, "np.ndarray"] = {}  # type: ignore
        self._source_texts: Dict[int, str] = {}

    @property
    def is_semantic_available(self) -> bool:
        """Check if semantic matching is available."""
        return self.embedder.available and USE_SEMANTIC_MATCHING

    def precompute_sources(self, sources: Sequence[Mapping[str, object]]) -> int:
        """Pre-compute embeddings for all sources.

        Args:
            sources: List of source dictionaries

        Returns:
            Number of sources successfully embedded
        """
        if not self.is_semantic_available:
            return 0

        count = 0
        for i, source in enumerate(sources):
            text = self._source_text(source)
            self._source_texts[i] = text
            embedding = self.embedder.embed(text)
            if embedding is not None:
                self._source_embeddings[i] = embedding
                count += 1

        logger.debug("Precomputed embeddings for %d/%d sources", count, len(sources))
        return count

    def find_best_match(
        self,
        finding: str,
        sources: Sequence[Mapping[str, object]],
        cited_urls: Optional[Set[str]] = None,
    ) -> Tuple[Optional[Mapping[str, object]], float]:
        """Find best matching source for a finding.

        Uses semantic similarity if available, otherwise falls back
        to keyword matching.

        B35.3: when multiple sources cluster within ``_AUTHORITY_TIE_EPSILON``
        of the top similarity score, the source with the higher
        ``authority_score`` (set by B35.1) wins. Authority is a tie-breaker;
        the threshold-based gate is unchanged.

        Args:
            finding: The finding text to match
            sources: List of source dictionaries
            cited_urls: Optional set of already-cited URLs to skip

        Returns:
            Tuple of (best_source, similarity_score)
            Returns (None, 0.0) if no match above threshold
        """
        cited_urls = cited_urls or set()

        if not self.is_semantic_available or not self._source_embeddings:
            # Fallback to keyword matching
            return self._keyword_fallback(finding, sources, cited_urls)

        # Embed finding
        finding_embedding = self.embedder.embed(finding)
        if finding_embedding is None:
            return self._keyword_fallback(finding, sources, cited_urls)

        # Score every eligible source so we can authority-tie-break the top.
        scored: List[Tuple[float, float, int]] = []
        for idx, source_embedding in self._source_embeddings.items():
            if not (0 <= idx < len(sources)):
                continue
            source = sources[idx]
            url = str(source.get("url", ""))
            if url and url in cited_urls:
                continue

            score = cosine_similarity(finding_embedding, source_embedding)
            scored.append((score, _authority_for(source), idx))

        if not scored:
            return self._keyword_fallback(finding, sources, cited_urls)

        # Pick the highest-similarity bucket, then break ties on authority,
        # then on original index for stability.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top_score = scored[0][0]
        eligible = [
            entry for entry in scored if top_score - entry[0] <= _AUTHORITY_TIE_EPSILON
        ]
        eligible.sort(key=lambda item: (-item[1], item[2]))
        best_score, _, best_idx = eligible[0]

        if best_score >= self.threshold and 0 <= best_idx < len(sources):
            return sources[best_idx], best_score

        # If semantic matching found nothing above threshold, try keyword fallback
        if best_score < self.threshold:
            return self._keyword_fallback(finding, sources, cited_urls)

        return None, 0.0

    def _source_text(self, source: Mapping[str, object]) -> str:
        """Extract text from source for embedding."""
        parts = []
        title = source.get("title")
        if title:
            parts.append(str(title))
        snippet = source.get("snippet")
        if snippet:
            parts.append(str(snippet))
        return " ".join(parts)

    def _keyword_fallback(
        self,
        finding: str,
        sources: Sequence[Mapping[str, object]],
        cited_urls: Set[str],
    ) -> Tuple[Optional[Mapping[str, object]], float]:
        """Fallback to keyword matching when embeddings unavailable.

        Args:
            finding: The finding text
            sources: List of source dictionaries
            cited_urls: Set of already-cited URLs to skip

        Returns:
            Tuple of (best_source, normalized_score)
        """
        finding_lower = finding.lower()
        finding_words = set(self._extract_keywords(finding_lower))
        best_score = 0
        best_source: Optional[Mapping[str, object]] = None

        for source in sources:
            url = str(source.get("url", ""))
            if url and url in cited_urls:
                continue

            score = 0
            title = str(source.get("title", "")).lower()
            title_words = set(self._extract_keywords(title))
            common = title_words & finding_words
            score += len(common) * 2

            snippet = str(source.get("snippet", "")).lower()
            snippet_words = set(self._extract_keywords(snippet))
            common_snippet = snippet_words & finding_words
            score += len(common_snippet)

            if score > best_score:
                best_score = score
                best_source = source

        # Normalize keyword score to 0-1 range (approx)
        normalized = min(best_score / 10.0, 1.0)

        if best_score >= 2:
            return best_source, normalized

        return None, 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text, filtering stopwords."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "it", "its", "as", "if", "then", "than",
            "so", "no", "not", "only", "just", "also", "more", "most", "some",
            "any", "all", "each", "every", "both", "few", "many", "much", "such",
            "very", "too", "quite", "rather", "about", "into", "over", "under",
        }
        words = []
        for word in text.split():
            clean = "".join(c for c in word if c.isalnum())
            if clean and len(clean) > 2 and clean not in stopwords:
                words.append(clean)
        return words


# Convenience function for one-off matching
def find_semantic_match(
    finding: str,
    sources: Sequence[Mapping[str, object]],
    cited_urls: Optional[Set[str]] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[Optional[Mapping[str, object]], float]:
    """Find the best matching source for a finding.

    Convenience function that creates a SemanticMatcher, precomputes
    source embeddings, and finds the best match.

    Args:
        finding: The finding text to match
        sources: List of source dictionaries
        cited_urls: Optional set of already-cited URLs to skip
        threshold: Minimum similarity threshold

    Returns:
        Tuple of (best_source, similarity_score)
    """
    matcher = SemanticMatcher(similarity_threshold=threshold)
    matcher.precompute_sources(sources)
    return matcher.find_best_match(finding, sources, cited_urls)


__all__ = [
    "OllamaEmbedder",
    "SemanticMatcher",
    "EmbeddingResult",
    "cosine_similarity",
    "find_semantic_match",
    "DEFAULT_MODEL",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "USE_SEMANTIC_MATCHING",
]
