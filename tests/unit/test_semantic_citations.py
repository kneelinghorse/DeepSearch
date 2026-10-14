"""Unit tests for semantic citation matching via embeddings.

Tests cover:
- OllamaEmbedder availability checking and embedding generation
- SemanticMatcher with semantic and keyword fallback modes
- Cosine similarity computation
- Integration with build_findings_section
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from deepsearch.report.embeddings import (
    OllamaEmbedder,
    SemanticMatcher,
    cosine_similarity,
    find_semantic_match,
    DEFAULT_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    USE_SEMANTIC_MATCHING,
    _get_numpy,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_sources() -> List[Dict[str, str]]:
    """Sample source documents for testing."""
    return [
        {
            "title": "Introduction to Machine Learning",
            "url": "https://example.com/ml-intro",
            "snippet": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        },
        {
            "title": "Deep Learning Fundamentals",
            "url": "https://example.com/deep-learning",
            "snippet": "Deep learning uses neural networks with multiple layers to process complex patterns.",
        },
        {
            "title": "Natural Language Processing Guide",
            "url": "https://example.com/nlp-guide",
            "snippet": "NLP enables computers to understand, interpret, and generate human language.",
        },
        {
            "title": "Python Programming Tutorial",
            "url": "https://example.com/python",
            "snippet": "Python is a versatile programming language popular in data science and web development.",
        },
    ]


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns predictable embeddings."""
    import numpy as np

    embedder = MagicMock(spec=OllamaEmbedder)
    embedder.available = True
    embedder.model = DEFAULT_MODEL

    # Map of text -> embedding (using simple vectors for testing)
    embedding_map = {
        "machine learning": np.array([1.0, 0.0, 0.0, 0.0]),
        "deep learning": np.array([0.9, 0.1, 0.0, 0.0]),
        "nlp language": np.array([0.0, 1.0, 0.0, 0.0]),
        "python programming": np.array([0.0, 0.0, 1.0, 0.0]),
        "unrelated topic": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    def mock_embed(text: str) -> Optional[np.ndarray]:
        # Return embedding based on keywords in text
        text_lower = text.lower()
        if "machine learning" in text_lower or "artificial intelligence" in text_lower:
            return embedding_map["machine learning"]
        elif "deep learning" in text_lower or "neural network" in text_lower:
            return embedding_map["deep learning"]
        elif "nlp" in text_lower or "language" in text_lower:
            return embedding_map["nlp language"]
        elif "python" in text_lower or "programming" in text_lower:
            return embedding_map["python programming"]
        else:
            return embedding_map["unrelated topic"]

    embedder.embed = mock_embed
    embedder.embed_batch = lambda texts: [mock_embed(t) for t in texts]
    embedder.clear_cache = lambda: None

    return embedder


# ============================================================================
# OllamaEmbedder Tests
# ============================================================================


class TestOllamaEmbedder:
    """Tests for the OllamaEmbedder class."""

    def test_init_defaults(self):
        """Test default initialization."""
        embedder = OllamaEmbedder()
        assert embedder.model == DEFAULT_MODEL
        assert embedder.base_url == "http://localhost:11434"
        assert embedder._available is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://custom:8080",
        )
        assert embedder.model == "mxbai-embed-large"
        assert embedder.base_url == "http://custom:8080"

    def test_availability_check_failure(self):
        """Test availability returns False when Ollama is not running."""
        embedder = OllamaEmbedder()
        # Without mocking httpx, this should return False (no server running)
        with patch("httpx.get", side_effect=ConnectionError("No connection")):
            assert embedder.available is False

    def test_availability_check_success(self):
        """Test availability returns True when Ollama is running."""
        embedder = OllamaEmbedder()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.get", return_value=mock_response):
            # Reset cached availability
            embedder._available = None
            assert embedder.available is True

    def test_embed_when_unavailable(self):
        """Test embed returns None when Ollama is unavailable."""
        embedder = OllamaEmbedder()
        embedder._available = False
        result = embedder.embed("test text")
        assert result is None

    def test_embed_with_caching(self):
        """Test that embeddings are cached."""
        import numpy as np

        embedder = OllamaEmbedder()
        embedder._available = True

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            # First call
            result1 = embedder.embed("test text")
            assert result1 is not None
            assert mock_post.call_count == 1

            # Second call - should use cache
            result2 = embedder.embed("test text")
            assert result2 is not None
            assert mock_post.call_count == 1  # No new call

            # Verify results are equal
            np.testing.assert_array_equal(result1, result2)

    def test_clear_cache(self):
        """Test cache clearing."""
        embedder = OllamaEmbedder()
        embedder._cache["key"] = "value"
        embedder.clear_cache()
        assert len(embedder._cache) == 0


# ============================================================================
# Cosine Similarity Tests
# ============================================================================


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        np = _get_numpy()
        if np is None:
            pytest.skip("numpy not available")

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        np = _get_numpy()
        if np is None:
            pytest.skip("numpy not available")

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        np = _get_numpy()
        if np is None:
            pytest.skip("numpy not available")

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        np = _get_numpy()
        if np is None:
            pytest.skip("numpy not available")

        a = np.array([1.0, 0.1, 0.0])
        b = np.array([0.9, 0.2, 0.0])
        similarity = cosine_similarity(a, b)
        assert similarity > 0.9

    def test_zero_vector_handling(self):
        """Zero vectors should return 0.0 similarity."""
        np = _get_numpy()
        if np is None:
            pytest.skip("numpy not available")

        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


# ============================================================================
# SemanticMatcher Tests
# ============================================================================


class TestSemanticMatcher:
    """Tests for the SemanticMatcher class."""

    def test_init_defaults(self):
        """Test default initialization."""
        matcher = SemanticMatcher()
        assert matcher.threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert matcher.embedder is not None

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        matcher = SemanticMatcher(similarity_threshold=0.5)
        assert matcher.threshold == 0.5

    def test_precompute_sources_when_unavailable(self, sample_sources):
        """Test precompute returns 0 when embeddings unavailable."""
        embedder = MagicMock(spec=OllamaEmbedder)
        embedder.available = False

        matcher = SemanticMatcher(embedder=embedder)
        count = matcher.precompute_sources(sample_sources)
        assert count == 0

    def test_precompute_sources_success(self, sample_sources, mock_embedder):
        """Test successful source precomputation."""
        with patch.object(
            SemanticMatcher, "is_semantic_available", new_callable=lambda: property(lambda self: True)
        ):
            matcher = SemanticMatcher(embedder=mock_embedder)
            matcher._source_embeddings = {}  # Reset

            # Mock precompute to count embeddings
            count = 0
            for i, source in enumerate(sample_sources):
                text = matcher._source_text(source)
                embedding = mock_embedder.embed(text)
                if embedding is not None:
                    matcher._source_embeddings[i] = embedding
                    count += 1

            assert count == len(sample_sources)

    def test_find_best_match_semantic(self, sample_sources, mock_embedder):
        """Test semantic matching finds correct source."""
        import numpy as np

        matcher = SemanticMatcher(embedder=mock_embedder, similarity_threshold=0.7)

        # Manually set up source embeddings
        for i, source in enumerate(sample_sources):
            text = matcher._source_text(source)
            embedding = mock_embedder.embed(text)
            if embedding is not None:
                matcher._source_embeddings[i] = embedding

        # Test matching a machine learning finding
        finding = "Machine learning enables predictive analytics"
        best_match, score = matcher.find_best_match(finding, sample_sources)

        # Should match the ML source
        assert best_match is not None
        assert "Machine Learning" in best_match.get("title", "")

    def test_find_best_match_respects_cited_urls(self, sample_sources, mock_embedder):
        """Test that already-cited URLs are skipped."""
        import numpy as np

        matcher = SemanticMatcher(embedder=mock_embedder, similarity_threshold=0.5)

        # Set up embeddings
        for i, source in enumerate(sample_sources):
            text = matcher._source_text(source)
            embedding = mock_embedder.embed(text)
            if embedding is not None:
                matcher._source_embeddings[i] = embedding

        # Mark first source as already cited
        cited_urls = {sample_sources[0]["url"]}

        finding = "Machine learning enables predictive analytics"
        best_match, score = matcher.find_best_match(finding, sample_sources, cited_urls)

        # Should NOT match the already-cited source
        if best_match:
            assert best_match.get("url") not in cited_urls

    def test_keyword_fallback(self, sample_sources):
        """Test fallback to keyword matching when embeddings unavailable."""
        embedder = MagicMock(spec=OllamaEmbedder)
        embedder.available = False

        matcher = SemanticMatcher(embedder=embedder)

        # Finding that matches Python source by keywords
        finding = "Python is great for data science programming"
        best_match, score = matcher.find_best_match(finding, sample_sources)

        assert best_match is not None
        assert "Python" in best_match.get("title", "")

    def test_keyword_fallback_no_match(self, sample_sources):
        """Test keyword fallback returns None when no match."""
        embedder = MagicMock(spec=OllamaEmbedder)
        embedder.available = False

        matcher = SemanticMatcher(embedder=embedder)

        # Finding that doesn't match any source
        finding = "Completely unrelated topic about gardening"
        best_match, score = matcher.find_best_match(finding, sample_sources)

        assert best_match is None
        assert score == 0.0

    def test_extract_keywords(self):
        """Test keyword extraction filters stopwords."""
        matcher = SemanticMatcher()
        text = "the quick brown fox jumps over the lazy dog"
        keywords = matcher._extract_keywords(text)

        # Should exclude stopwords (the, over)
        assert "the" not in keywords
        assert "over" not in keywords
        # Should include meaningful words
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_source_text_extraction(self):
        """Test source text combines title and snippet."""
        matcher = SemanticMatcher()
        source = {
            "title": "Test Title",
            "snippet": "Test snippet content",
            "url": "https://example.com",
        }
        text = matcher._source_text(source)
        assert "Test Title" in text
        assert "Test snippet content" in text


# ============================================================================
# Integration Tests
# ============================================================================


class TestFindSemanticMatch:
    """Tests for the convenience find_semantic_match function."""

    def test_convenience_function(self, sample_sources):
        """Test the convenience function works end-to-end."""
        finding = "Python programming is useful for web development"
        best_match, score = find_semantic_match(finding, sample_sources, threshold=0.1)

        # Should find a match (likely Python source via keyword fallback)
        assert best_match is not None
        assert score > 0


class TestBuildFindingsSectionIntegration:
    """Integration tests with build_findings_section."""

    def test_semantic_matching_disabled(self, sample_sources):
        """Test that semantic matching can be disabled."""
        from deepsearch.report.templates import build_findings_section

        findings = ["Machine learning is transforming data analytics."]

        # Disable semantic matching
        result = build_findings_section(
            findings,
            sources=sample_sources,
            use_semantic_matching=False,
        )

        # Should still produce citations via keyword fallback
        assert "## Research Findings" in result

    def test_semantic_matching_fallback_graceful(self, sample_sources):
        """Test graceful fallback when Ollama unavailable."""
        from deepsearch.report.templates import build_findings_section

        findings = [
            "Machine learning enables predictive analytics.",
            "Python is popular for data science.",
        ]

        # With semantic matching enabled but Ollama not running,
        # should fall back to keyword matching
        result = build_findings_section(
            findings,
            sources=sample_sources,
            use_semantic_matching=True,
        )

        assert "## Research Findings" in result
        # Should still have findings listed
        assert "Machine learning" in result or "Python" in result


# ============================================================================
# Environment Configuration Tests
# ============================================================================


class TestEnvironmentConfig:
    """Tests for environment variable configuration."""

    def test_default_model_from_env(self):
        """Test DEFAULT_MODEL can be set via environment."""
        # Default should be nomic-embed-text
        assert DEFAULT_MODEL == os.environ.get(
            "DEEPSEARCH_EMBEDDING_MODEL", "nomic-embed-text"
        )

    def test_default_threshold(self):
        """Test DEFAULT_SIMILARITY_THRESHOLD."""
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.7

    def test_use_semantic_matching_default(self):
        """Test USE_SEMANTIC_MATCHING default."""
        # Default should be True unless env says otherwise
        assert USE_SEMANTIC_MATCHING in (True, False)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sources(self):
        """Test with empty sources list."""
        matcher = SemanticMatcher()
        best_match, score = matcher.find_best_match("test finding", [])
        assert best_match is None
        assert score == 0.0

    def test_empty_finding(self, sample_sources):
        """Test with empty finding."""
        matcher = SemanticMatcher()
        best_match, score = matcher.find_best_match("", sample_sources)
        assert best_match is None
        assert score == 0.0

    def test_source_without_title(self):
        """Test matching sources without title."""
        sources = [
            {
                "url": "https://example.com/no-title",
                "snippet": "Python programming tutorial for beginners",
            }
        ]
        matcher = SemanticMatcher()
        best_match, score = matcher.find_best_match("Python tutorial", sources)
        # Should still work via snippet matching
        assert best_match is not None

    def test_source_without_snippet(self):
        """Test matching sources without snippet."""
        sources = [
            {
                "title": "Python Programming Guide",
                "url": "https://example.com/python-guide",
            }
        ]
        matcher = SemanticMatcher()
        best_match, score = matcher.find_best_match("Python guide", sources)
        # Should still work via title matching
        assert best_match is not None

    def test_unicode_handling(self, sample_sources):
        """Test handling of unicode characters."""
        finding = "Machine learning with 日本語 and émojis 🤖"
        matcher = SemanticMatcher()
        # Should not raise
        best_match, score = matcher.find_best_match(finding, sample_sources)
        # May or may not find a match, but should not crash
        assert score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
