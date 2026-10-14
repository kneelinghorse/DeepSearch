"""S69 FActScore Local Eval Harness — Part 1.

Local-only evaluation tooling. Consumes cached mission outputs (protocol
JSON + sources sidecar + sibling stripped markdown) and produces a
FActScore per mission. Local-only; never wired into the production runtime.
S70 wires the harness into the daily-eval cadence.

Per ``cmos/reports/sprint_69_approach_and_plan.md`` PD1–PD10 (locked
2026-04-28):

* PD1 — STRONG, claim-level entailment via DeepEval ``FaithfulnessMetric``.
* PD2 — LLM-driven decomposition; verification + decomposition models are
  recorded on the report (``deepseek-v4-pro`` per the locked S62 stack).
* PD3 — Probe = ``FaithfulnessMetric``; default threshold ``0.5``.
* PD4 — Probe input = source body string from ``sources_collected``.
  When the cited URL has no body in the sidecar (or the protocol's
  embedded ``sources_collected``), the citation is classified ``unknown``
  — the same separation as S68's ``unknown`` vs ``dead`` distinction.
  ``CitationAnnotation.live`` filters the probe pool to ``live==True``
  per S68 critique rule 8.
* PD5 — Thin wrapper over DeepEval. Per-citation probe via ``measure``
  inside ``asyncio.to_thread``; outer ``asyncio.wait`` enforces a
  wall-clock budget. Partial returns flag ``budget_exhausted=True``.
* PD6 — No mutation of ``CitationAnnotation``. Output is the
  ``FActScoreReport`` dataclass; the m02 CLI persists it to
  ``checkpoints/factscore/<mission_id>.json``.
* PD7 — ``FActScoreReport`` is JSON-serialisable;
  ``unsupported_count_per_citation`` is the load-bearing FActScore-axis
  counterpart to S68's ``hallucinated_url_count``.
* PD10 — No imports from any production-runtime surface
  (``codeagent_finalize_node``, ``citation_liveness``, ``critique``,
  ``contract_to_prompt``, ``codeagent_runner``, ``worker.writer``, or any
  ``deepsearch.agent.tools.*``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from deepsearch.data_paths import data_path

logger = logging.getLogger(__name__)


HARNESS_VERSION: str = "s69-1.0"
"""Harness build identifier; bumps with every schema change to FActScoreReport."""

DEFAULT_BUDGET_SECONDS: float = 300.0
"""PD5 — 5 minute wall-clock cap per mission scoring."""

DEFAULT_DECOMPOSITION_MODEL: str = "deepseek-v4-pro"
"""PD2 — locked S62 production binding."""

DEFAULT_VERIFICATION_MODEL: str = "deepseek-v4-pro"
"""PD3 — locked S62 production binding."""

DEFAULT_THRESHOLD: float = 0.5
"""PD3 — FaithfulnessMetric pass/fail threshold (FActScore paper default)."""


# ---------------------------------------------------------------------------
# PD7 — FActScoreReport dataclass.
# ---------------------------------------------------------------------------


@dataclass
class FActScoreReport:
    """Harness output. JSON-serialisable.

    * ``mission_id`` — cached mission identifier.
    * ``factscore`` — ``supported_count / decomposed_claim_count``; ``0.0``
      when there is nothing to score.
    * ``decomposed_claim_count`` — total atomic claims atomised across all
      probed citations (plus one per missing-body citation, classified
      ``unknown``).
    * ``supported_count`` — atomic claims with verdict ``"yes"``.
    * ``unsupported_count`` — atomic claims with verdict ``"no"``.
    * ``unknown_count`` — atomic claims with verdict ``"idk"``, citations
      whose source body was unavailable, and citations cancelled by the
      budget.
    * ``unsupported_count_per_citation`` — Dict[url, count] of unsupported
      atomic claims per citation. Load-bearing PD7 field; the FActScore-
      axis counterpart to S68's ``hallucinated_url_count``.
    * ``wall_clock_seconds`` — total elapsed scoring time.
    * ``budget_exhausted`` — True if any probe was cancelled by the
      wall-clock cutoff.
    * ``decomposition_model_id`` / ``verification_model_id`` — the model
      strings as requested at call time.
    * ``threshold`` — ``FaithfulnessMetric`` threshold the probe used.
    * ``harness_version`` — ``HARNESS_VERSION``.
    """

    mission_id: str
    factscore: float
    decomposed_claim_count: int
    supported_count: int
    unsupported_count: int
    unknown_count: int
    unsupported_count_per_citation: Dict[str, int]
    wall_clock_seconds: float
    budget_exhausted: bool
    decomposition_model_id: str
    verification_model_id: str
    threshold: float
    harness_version: str = HARNESS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict; the m02 CLI writes this verbatim to disk."""
        return {
            "mission_id": str(self.mission_id),
            "factscore": float(self.factscore),
            "decomposed_claim_count": int(self.decomposed_claim_count),
            "supported_count": int(self.supported_count),
            "unsupported_count": int(self.unsupported_count),
            "unknown_count": int(self.unknown_count),
            "unsupported_count_per_citation": {
                str(url): int(count)
                for url, count in self.unsupported_count_per_citation.items()
            },
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "budget_exhausted": bool(self.budget_exhausted),
            "decomposition_model_id": str(self.decomposition_model_id),
            "verification_model_id": str(self.verification_model_id),
            "threshold": float(self.threshold),
            "harness_version": str(self.harness_version),
        }


# ---------------------------------------------------------------------------
# DeepEval lazy-import indirection. Tests monkeypatch this to inject fakes
# without requiring the real ``deepeval`` to be installed.
# ---------------------------------------------------------------------------


def _resolve_deepeval() -> Tuple[Any, Any]:
    """Return ``(FaithfulnessMetric, LLMTestCase)`` from DeepEval.

    Lazy so the harness module imports cleanly when DeepEval is not
    installed (which is fine for the AST-guard tests + the persistence
    round-trip integration test). Raises ``RuntimeError`` only when the
    caller actually needs to score.
    """
    try:
        from deepeval.metrics import FaithfulnessMetric  # type: ignore
        from deepeval.test_case import LLMTestCase  # type: ignore
    except ImportError as exc:  # pragma: no cover — dependency guard
        raise RuntimeError(
            "deepeval is not installed; run `pip install deepeval` to enable "
            "the FActScore harness"
        ) from exc
    return FaithfulnessMetric, LLMTestCase


def _resolve_verification_llm(verification_model: str) -> Any:
    """Route the harness's ``verification_model`` string through the S71
    Path B adapter factory.

    Returns either a :class:`DeepEvalBaseLLM` adapter instance (for the
    locked DeepSeek + Gemini providers) or the raw string (preserves
    DeepEval's default string-resolution path for any other name).

    Indirection lives here — not inline at the
    ``FaithfulnessMetric(model=…)`` callsite — so the test harness can
    monkeypatch this single module-level symbol to inject a fake
    DeepEvalBaseLLM without depending on the real ``deepeval`` /
    ``openai`` / ``google.genai`` SDKs.
    """

    from deepsearch.eval.deepeval_adapter import resolve_deepeval_llm  # local import

    return resolve_deepeval_llm(verification_model)


# ---------------------------------------------------------------------------
# Cache discovery + I/O helpers.
# ---------------------------------------------------------------------------


def resolve_protocol_path(
    mission_id: str,
    search_dirs: Optional[Tuple[Path, ...]] = None,
) -> Path:
    """Locate a cached ``<mission_id>*_protocol.json`` by prefix match.

    Walks the candidate search dirs in order and returns the first match.
    The locked protocol filename convention from
    ``deepsearch.report.generator.save_protocol`` is
    ``<sanitised_mission_id>_<title_slug>_protocol.json``; this helper
    matches on the ``<mission_id>_`` prefix to tolerate the title-slug
    suffix.

    Args:
        mission_id: The mission identifier as recorded by the runner.
        search_dirs: Override the default candidate list (mostly for
            tests). When ``None``, searches protocols/ and reports/ under
            the data root, with ``DEEPSEARCH_REPORT_DIR`` prepended if set.

    Returns:
        The first matching ``Path``.

    Raises:
        FileNotFoundError: When no candidate file exists; the message
            enumerates every directory walked so the caller can act on
            the gap (typically by passing ``--protocol`` directly).
    """
    import os

    candidates: List[Path]
    if search_dirs is not None:
        candidates = list(search_dirs)
    else:
        candidates = [data_path("protocols"), data_path("reports")]
        override = os.getenv("DEEPSEARCH_REPORT_DIR")
        if override:
            candidates.insert(0, Path(override))

    sanitised = mission_id.strip()
    if not sanitised:
        raise FileNotFoundError("mission_id is empty; cannot resolve protocol path")

    walked: List[str] = []
    for directory in candidates:
        walked.append(str(directory))
        if not directory.exists() or not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            if not entry.name.endswith("_protocol.json"):
                continue
            if entry.name.startswith(f"{sanitised}_"):
                return entry

    raise FileNotFoundError(
        f"No cached protocol found for mission_id={sanitised!r}. "
        f"Searched (in order): {walked}. "
        f"Provide --protocol <path> to bypass discovery."
    )


def _resolve_markdown_path(protocol_path: Path) -> Path:
    """Map a protocol JSON path to its sibling stripped-markdown file.

    Convention from ``deepsearch.report.generator.save_protocol`` /
    ``save_report``: ``<stem>_protocol.json`` sits alongside ``<stem>.md``.
    Returns the candidate path even if it does not exist; the caller
    treats a missing markdown file as all-unknown (every live citation
    becomes an ``unknown`` atomic claim).
    """
    stem = protocol_path.stem
    if stem.endswith("_protocol"):
        stem = stem[: -len("_protocol")]
    return protocol_path.with_name(f"{stem}.md")


def _load_protocol(protocol_path: Path) -> Dict[str, Any]:
    raw = protocol_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"protocol at {protocol_path} is not a JSON object (got {type(data).__name__})"
        )
    return data


def _load_sources(
    sources_path: Optional[Path],
    protocol: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Resolve ``sources_collected`` from sidecar JSON or the protocol body.

    The runner's ``sources_collected`` accumulator (S64) is not currently
    embedded on ``MissionProtocolComplete``; the m02 CLI accepts a
    sidecar path. PD4 risk row anticipates this: when the body cannot be
    located, the harness reports the citation as ``unknown``.
    """
    if sources_path is not None:
        try:
            data = json.loads(sources_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("sources_path %s not found; falling back to protocol body", sources_path)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("sources_path %s failed to load: %r", sources_path, exc)
        else:
            if isinstance(data, list):
                return [s for s in data if isinstance(s, Mapping)]
            if isinstance(data, Mapping):
                embedded = data.get("sources_collected") or data.get("sources")
                if isinstance(embedded, list):
                    return [s for s in embedded if isinstance(s, Mapping)]

    embedded = protocol.get("sources_collected")
    if isinstance(embedded, list):
        return [s for s in embedded if isinstance(s, Mapping)]
    return []


def _body_of(source: Mapping[str, Any]) -> str:
    """Return the first non-empty body field on a source entry.

    Tries ``body`` → ``text`` → ``content`` → ``snippet``. The runner
    accumulates a 300-char ``snippet`` from ``source_fetch``; longer
    bodies appear under ``text`` or ``content`` if a future runner
    persists them. Empty string when nothing is present.
    """
    for key in ("body", "text", "content", "snippet"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _claim_chunk(markdown_text: str, start_index: int, end_index: int) -> str:
    """Extract a citation's claim chunk by codepoint slice; clamped safely."""
    if not markdown_text:
        return ""
    start = max(0, int(start_index))
    end = max(start, int(end_index))
    return markdown_text[start:end]


# ---------------------------------------------------------------------------
# Public entry-point.
# ---------------------------------------------------------------------------


async def score_mission(
    mission_id: str,
    protocol_path: Path,
    sources_path: Optional[Path] = None,
    *,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    decomposition_model: str = DEFAULT_DECOMPOSITION_MODEL,
    verification_model: str = DEFAULT_VERIFICATION_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> FActScoreReport:
    """Score one cached mission's claim-level faithfulness.

    Loads the cached protocol JSON + sibling stripped markdown + optional
    sources sidecar. Filters citations to ``live==True`` (PD4 + S68
    critique rule 8) and probes each via DeepEval's ``FaithfulnessMetric``.
    Per ``(claim_chunk, source_body)`` pair: atomise → entail per atomic
    claim → tally. Returns a :class:`FActScoreReport`. Never raises;
    outer ``try/except`` returns a populated empty/partial report on any
    unexpected error.

    Args:
        mission_id: Cached mission identifier (used as the
            ``LLMTestCase.input`` field and on the report).
        protocol_path: Path to the cached protocol JSON. Markdown is
            resolved by convention from this path (``<stem>.md``).
        sources_path: Optional sidecar path to ``sources_collected`` (the
            runner accumulator). When ``None``, the harness looks for
            ``sources_collected`` embedded in the protocol body. Either
            way, missing source bodies classify their citations as
            ``unknown``.
        budget_seconds: PD5 wall-clock budget. Defaults to
            :data:`DEFAULT_BUDGET_SECONDS`.
        decomposition_model: Recorded on the report. PD2.
        verification_model: Passed to ``FaithfulnessMetric(model=…)``.
            PD3.
        threshold: ``FaithfulnessMetric`` pass/fail threshold. PD3.

    Returns:
        A populated :class:`FActScoreReport`.
    """

    start = time.monotonic()

    def _empty(*, budget_exhausted: bool = False, unknown_count: int = 0) -> FActScoreReport:
        elapsed = time.monotonic() - start
        return FActScoreReport(
            mission_id=mission_id,
            factscore=0.0,
            decomposed_claim_count=unknown_count,
            supported_count=0,
            unsupported_count=0,
            unknown_count=unknown_count,
            unsupported_count_per_citation={},
            wall_clock_seconds=elapsed,
            budget_exhausted=budget_exhausted,
            decomposition_model_id=decomposition_model,
            verification_model_id=verification_model,
            threshold=threshold,
        )

    try:
        protocol = _load_protocol(Path(protocol_path))
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.error("score_mission: protocol load failed for %s: %r", mission_id, exc)
        return _empty()

    citations_raw = protocol.get("citations") or []
    if not isinstance(citations_raw, list):
        return _empty()

    live_citations: List[Mapping[str, Any]] = [
        ann
        for ann in citations_raw
        if isinstance(ann, Mapping) and ann.get("live") is True
    ]
    if not live_citations:
        return _empty()

    sources = _load_sources(
        Path(sources_path) if sources_path is not None else None,
        protocol,
    )
    url_to_body: Dict[str, str] = {}
    for src in sources:
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        body = _body_of(src)
        if body and url not in url_to_body:
            url_to_body[url] = body

    markdown_path = _resolve_markdown_path(Path(protocol_path))
    markdown_text = ""
    if markdown_path.exists():
        try:
            markdown_text = markdown_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "score_mission: markdown load failed for %s at %s: %r",
                mission_id,
                markdown_path,
                exc,
            )

    try:
        FaithfulnessMetric, LLMTestCase = _resolve_deepeval()
    except RuntimeError as exc:
        logger.error("score_mission: deepeval unavailable for %s: %r", mission_id, exc)
        return _empty(unknown_count=len(live_citations))

    async def _probe_one(ann: Mapping[str, Any]) -> Tuple[int, int, int, int, str]:
        """Probe one citation; returns ``(atoms, supported, unsupported, unknown, url)``."""
        url = str(ann.get("url") or "").strip()
        body = url_to_body.get(url, "")
        chunk = _claim_chunk(
            markdown_text,
            int(ann.get("start_index") or 0),
            int(ann.get("end_index") or 0),
        )
        if not chunk or not body:
            # PD4 — missing body or empty cited span ⇒ one ``unknown`` atom.
            return 1, 0, 0, 1, url

        try:
            test_case = LLMTestCase(
                input=mission_id,
                actual_output=chunk,
                retrieval_context=[body],
            )
            metric = FaithfulnessMetric(
                threshold=threshold,
                model=_resolve_verification_llm(verification_model),
            )
            await asyncio.to_thread(metric.measure, test_case)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "score_mission: FaithfulnessMetric.measure raised for %s: %r",
                url,
                exc,
            )
            return 1, 0, 0, 1, url

        verdicts = list(getattr(metric, "verdicts", []) or [])
        if not verdicts:
            score_value = float(getattr(metric, "score", 0.0) or 0.0)
            if score_value >= threshold:
                return 1, 1, 0, 0, url
            return 1, 0, 1, 0, url

        s = u = k = 0
        for v in verdicts:
            verdict_str = str(getattr(v, "verdict", "") or "").strip().lower()
            if verdict_str == "yes":
                s += 1
            elif verdict_str == "no":
                u += 1
            else:
                k += 1
        return len(verdicts), s, u, k, url

    tasks: List[asyncio.Task[Tuple[int, int, int, int, str]]] = [
        asyncio.create_task(_probe_one(ann)) for ann in live_citations
    ]

    elapsed = time.monotonic() - start
    remaining = max(0.0, budget_seconds - elapsed)

    decomposed = supported = unsupported = unknown = 0
    per_citation: Dict[str, int] = {}
    budget_exhausted = False

    try:
        done, pending = await asyncio.wait(tasks, timeout=remaining)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.error("score_mission: outer asyncio.wait raised: %r", exc)
        for t in tasks:
            t.cancel()
        return _empty(unknown_count=len(live_citations))

    for task in done:
        try:
            atoms, s_count, u_count, k_count, url = task.result()
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("score_mission: probe task raised: %r", exc)
            decomposed += 1
            unknown += 1
            continue
        decomposed += atoms
        supported += s_count
        unsupported += u_count
        unknown += k_count
        if u_count > 0 and url:
            per_citation[url] = per_citation.get(url, 0) + u_count

    if pending:
        budget_exhausted = True
        for task in pending:
            task.cancel()
            decomposed += 1
            unknown += 1
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception:  # pragma: no cover — defence in depth
            pass

    factscore_value = (float(supported) / float(decomposed)) if decomposed > 0 else 0.0
    elapsed = time.monotonic() - start

    return FActScoreReport(
        mission_id=mission_id,
        factscore=factscore_value,
        decomposed_claim_count=decomposed,
        supported_count=supported,
        unsupported_count=unsupported,
        unknown_count=unknown,
        unsupported_count_per_citation=per_citation,
        wall_clock_seconds=elapsed,
        budget_exhausted=budget_exhausted,
        decomposition_model_id=decomposition_model,
        verification_model_id=verification_model,
        threshold=threshold,
    )


__all__ = [
    "DEFAULT_BUDGET_SECONDS",
    "DEFAULT_DECOMPOSITION_MODEL",
    "DEFAULT_THRESHOLD",
    "DEFAULT_VERIFICATION_MODEL",
    "FActScoreReport",
    "HARNESS_VERSION",
    "resolve_protocol_path",
    "score_mission",
]


# ---------------------------------------------------------------------------
# CLI shim — supports ``python -m deepsearch.eval.factscore <args>`` per the
# m02 plan-doc command form. The actual argparse + asyncio plumbing lives
# in ``deepsearch.eval.__main__`` so ``python -m deepsearch.eval`` also
# works.
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover — invoked via python -m
    from deepsearch.eval.__main__ import main as _cli_main

    raise SystemExit(_cli_main())
