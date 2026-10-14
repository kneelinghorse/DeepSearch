"""Tool integrations (web search, local files, RAG) used by the agent."""

from .arxiv_search import (
    ArxivResult,
    ArxivSearchClient,
    ArxivSearchError,
)
from .openalex_search import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexResult,
)
from .semantic_scholar_search import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarResult,
)
from .linkup_search import (
    LinkupResult,
    LinkupSearchClient,
    LinkupSearchError,
    LinkupSearchInput,
    LinkupSearchTool,
)
from .exa_search import (
    ExaResult,
    ExaSearchClient,
    ExaSearchError,
    ExaSearchInput,
    ExaSearchTool,
)
from .google_search import (
    GoogleSearchClient,
    GoogleSearchError,
    GoogleSearchInput,
    GoogleSearchResult,
    GoogleSearchTool,
    GroundingChunk,
    GroundingSupport,
    extract_grounded_citations,
)
from .serp_search import (
    SerpResult,
    SerpSearchClient,
    SerpSearchError,
    SerpSearchInput,
    SerpSearchTool,
)
from .web_search import (
    TavilyResult,
    TavilySearchClient,
    TavilySearchInput,
    TavilySearchTool,
    WebSearchError,
)

__all__ = [
    # arXiv native search (B24.3: academic lane for Deep/Alpha tiers)
    "ArxivResult",
    "ArxivSearchClient",
    "ArxivSearchError",
    # OpenAlex native search (B24.4: academic lane with publisher URLs)
    "OpenAlexClient",
    "OpenAlexError",
    "OpenAlexResult",
    # Semantic Scholar native search (B24.4: academic lane with publisher URLs)
    "SemanticScholarClient",
    "SemanticScholarError",
    "SemanticScholarResult",
    # Exa Search (semantic/embedding-based)
    "ExaResult",
    "ExaSearchClient",
    "ExaSearchError",
    "ExaSearchInput",
    "ExaSearchTool",
    # Google Search (default)
    "GoogleSearchClient",
    "GoogleSearchError",
    "GoogleSearchInput",
    "GoogleSearchResult",
    "GoogleSearchTool",
    "GroundingChunk",
    "GroundingSupport",
    "extract_grounded_citations",
    # Linkup Search (primary web search backend — replaces SerpAPI)
    "LinkupResult",
    "LinkupSearchClient",
    "LinkupSearchError",
    "LinkupSearchInput",
    "LinkupSearchTool",
    # SerpAPI (legacy: triple search for Alpha diversity — requires SERPAPI_API_KEY)
    "SerpResult",
    "SerpSearchClient",
    "SerpSearchError",
    "SerpSearchInput",
    "SerpSearchTool",
    # Tavily (fallback)
    "TavilyResult",
    "TavilySearchClient",
    "TavilySearchInput",
    "TavilySearchTool",
    "WebSearchError",
]
