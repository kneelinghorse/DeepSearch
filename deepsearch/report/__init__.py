"""Utilities for generating and formatting DeepSearch research reports."""

from .embeddings import (
    OllamaEmbedder,
    SemanticMatcher,
    cosine_similarity,
    find_semantic_match,
    DEFAULT_MODEL as EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    USE_SEMANTIC_MATCHING,
)
from .evidence import Evidence, deduplicate_by_url, generate_evidence_list
from .generator import (
    _build_executive_summary,
    generate_markdown_report,
    save_protocol,
    save_report,
)
from .protocol import (
    MissionProtocolComplete,
    QualityCheckpoint,
    REQUIRED_COMPLETION_GATES,
    generate_mission_protocol,
)
from .questions import KeyQuestion, generate_key_questions
from .synthesis import (
    ResearchStatement,
    Synthesis,
    generate_research_statement,
    generate_synthesis,
)
from .citation_assistant import (
    CitationAssistant,
    CitationSuggestion,
    OllamaClient,
    batch_suggest_citations,
    create_ollama_assistant,
)

__all__ = [
    "generate_markdown_report",
    "save_report",
    "save_protocol",
    "_build_executive_summary",
    "Evidence",
    "generate_evidence_list",
    "deduplicate_by_url",
    "KeyQuestion",
    "generate_key_questions",
    "ResearchStatement",
    "Synthesis",
    "generate_research_statement",
    "generate_synthesis",
    "MissionProtocolComplete",
    "QualityCheckpoint",
    "REQUIRED_COMPLETION_GATES",
    "generate_mission_protocol",
    # Embeddings/semantic matching
    "OllamaEmbedder",
    "SemanticMatcher",
    "cosine_similarity",
    "find_semantic_match",
    "EMBEDDING_MODEL",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "USE_SEMANTIC_MATCHING",
    # Citation assistant
    "CitationAssistant",
    "CitationSuggestion",
    "OllamaClient",
    "batch_suggest_citations",
    "create_ollama_assistant",
]
