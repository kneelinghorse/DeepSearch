"""S64 live-retrieval tool inventory for the CodeAgent.

Replaces the S62 fixed-pool tools (source_lookup, list_sources_by_authority_tier,
search_sources) with a live-retrieval surface so the CodeAgent owns retrieval
directly rather than receiving an admitted_pool from upstream LangGraph stages.

Plan reference: cmos/reports/sprint_64_approach_and_plan.md.

Submodules:
- ``web_search``    — single-lane Linkup search.
- ``source_fetch``  — URL → page text with status taxonomy.
- ``url_liveness``  — HEAD/Range probe.
- ``scratchpad``    — closure-captured (write, read) pair, fresh per CodeAgent run.

Import the public callables directly from each submodule, e.g.
``from deepsearch.agent.tools.web_search import web_search``. We deliberately
do NOT re-export at the package level: each tool's public name matches its
module name, and re-exporting would shadow the module attribute and break
unit-test patches that target ``module.httpx`` / ``module.LinkupSearchClient``.
"""

from __future__ import annotations

from . import rerank, scratchpad, source_fetch, url_liveness, web_search

__all__ = ["rerank", "scratchpad", "source_fetch", "url_liveness", "web_search"]
