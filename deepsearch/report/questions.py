"""TraceLab KeyQuestion generation from AgentState mission objectives.

This module transforms mission_objectives from AgentState into TraceLab
KeyQuestion objects. Each objective becomes a research question with
matched findings as the answer and reflection scores as confidence.

S84 m04: a question is "answered" only when real findings back it — the
fabricated placeholder promotion was removed. TraceLab's INGEST API business
rules (the CLI --auto-upload path only; the production worker writes Postgres
directly) still document a >=1-answered requirement and need a server-side
relaxation before auto-upload is used — see cmos/planning/
deepsearch-integration.md and the s84-m04 CMOS completion record.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from deepsearch.agent.concept_clustering import summarize_state_findings
from deepsearch.agent.state import AgentState
from deepsearch.mission.contract import resolve_research_objectives


class KeyQuestion(BaseModel):
    """TraceLab-compatible KeyQuestion object.

    This model matches the TraceLab KeyQuestion schema for mission protocol
    ingestion. At least one question must be "answered" for validation.
    """

    question: str = Field(description="The research question")
    status: Literal["open", "researching", "answered"] = Field(
        default="open",
        description="Question status: open, researching, or answered",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Answer text (required if status='answered')",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    owner: Optional[str] = Field(
        default=None,
        description="Owner/assignee for the question",
    )

    @field_validator("answer")
    @classmethod
    def answer_required_when_answered(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        """Ensure answer is non-empty when status is 'answered'."""
        # Access status from the data being validated
        status = info.data.get("status") if hasattr(info, "data") else None
        if status == "answered" and not v:
            raise ValueError("answer is required when status is 'answered'")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(exclude_none=True)


def generate_key_questions(state: AgentState) -> List[KeyQuestion]:
    """Generate TraceLab KeyQuestion objects from mission objectives.

    Transforms each mission objective into a KeyQuestion with:
    - question: Objective text formatted as a question
    - answer: Matched findings concatenated
    - confidence: From reflection_scores.per_objective_scores
    - status: "answered" if answer is substantial, else "open"/"researching"

    Args:
        state: AgentState with mission_objectives, findings, reflection_scores

    Returns:
        List of KeyQuestion objects. A question is "answered" only when real
        findings back it (S84 m04 — no fabricated placeholder answers).
    """
    objectives = resolve_research_objectives(
        state.get("mission_contract"),
        fallback=state.get("mission_objectives") or [],
    )
    findings = list(summarize_state_findings(state).get("display_findings") or [])
    reflection_scores = state.get("reflection_scores") or {}
    per_objective = reflection_scores.get("per_objective_scores") or []

    # Handle empty objectives - generate a default question.
    # S84 m04 truth repair: 'answered' ONLY when real findings back the
    # answer. The pre-m04 branch always fabricated an answered question with
    # a placeholder answer and a dead-reflection confidence (0.5 constant on
    # every production mission).
    if not objectives:
        mission_id = state.get("mission_id", "research")
        default_answer = "\n".join(findings[:5]) if findings else None
        return [
            KeyQuestion(
                question=f"What are the key findings for {mission_id}?",
                status="answered" if default_answer else "open",
                answer=default_answer,
                confidence=None,
            )
        ]

    questions: List[KeyQuestion] = []
    has_answered = False

    for idx, objective in enumerate(objectives):
        objective_id = f"OBJ-{idx + 1}"

        # Format question
        question_text = _format_question(objective)

        # Match findings to objective
        matched_findings = _match_findings_to_objective(objective, findings)
        answer = "\n".join(matched_findings) if matched_findings else None

        # Get confidence from reflection scores
        confidence = _get_objective_confidence(objective_id, per_objective)

        # Determine status based on answer quality
        if answer and len(answer.strip()) > 20:
            status: Literal["open", "researching", "answered"] = "answered"
            has_answered = True
        elif matched_findings:
            status = "researching"
        else:
            status = "open"

        questions.append(
            KeyQuestion(
                question=question_text,
                status=status,
                answer=answer,
                confidence=confidence,
            )
        )

    # Ensure at least one is answered (TraceLab requirement)
    if not has_answered and questions:
        _ensure_one_answered(questions)

    return questions


def _ensure_one_answered(questions: List[KeyQuestion]) -> None:
    """Promote the first question with real content to "answered".

    Modifies the list in place. S84 m04 truth repair: when NO question has
    content, every question stays honestly open — the pre-m04 fallback
    force-wrote the placeholder 'Research conducted but specific findings
    pending synthesis.' with a fabricated confidence=0.3 onto question[0]
    of every production mission (state findings have no writer on the
    post-S64 codeagent path, so the placeholder shipped unconditionally).

    Args:
        questions: List of KeyQuestion objects to modify in place.
    """
    for idx, q in enumerate(questions):
        if q.answer:
            questions[idx] = KeyQuestion(
                question=q.question,
                status="answered",
                answer=q.answer,
                confidence=q.confidence,
                owner=q.owner,
            )
            return


def _format_question(objective: str) -> str:
    """Ensure objective is formatted as a question ending with '?'.

    Transforms objective statements into questions:
    - Already a question? Return unchanged
    - "Document X" -> "What are the X?"
    - "Analyze Y" -> "What is the analysis of Y?"
    - Other -> Append "?"

    Args:
        objective: Objective text string.

    Returns:
        Question string ending with '?'.
    """
    text = objective.strip()
    if not text:
        return "What was researched?"

    # Already a question
    if text.endswith("?"):
        return text

    # Convert statement to question based on leading verb
    lower = text.lower()
    if lower.startswith(("document ", "list ", "catalog ", "identify ")):
        # Find first space and extract the rest
        space_idx = text.find(" ")
        if space_idx > 0:
            rest = text[space_idx + 1 :]
            # Remove leading article if present to avoid "What are the the X?"
            rest_lower = rest.lower()
            if rest_lower.startswith(("the ", "a ", "an ")):
                article_space = rest.find(" ")
                if article_space > 0:
                    rest = rest[article_space + 1 :]
            return f"What are the {rest}?"
    elif lower.startswith(("analyze ", "evaluate ", "assess ")):
        space_idx = text.find(" ")
        if space_idx > 0:
            rest = text[space_idx + 1 :]
            return f"What is the analysis of {rest}?"

    # Default: just append question mark
    return f"{text}?"


def _match_findings_to_objective(
    objective: str,
    findings: List[str],
) -> List[str]:
    """Find findings related to an objective using keyword matching.

    Uses simple keyword extraction and matching. Future enhancement could
    use TF-IDF or embedding similarity for better semantic matching.

    Args:
        objective: Objective text to match against.
        findings: List of finding strings to search.

    Returns:
        List of matched findings (max 5). Empty list if no matches.
    """
    if not findings:
        return []

    # Extract keywords from objective (nouns, significant words)
    keywords = _extract_keywords(objective)
    if not keywords:
        # Fallback: return first 2 findings if we can't extract keywords
        return findings[:2]

    matched: List[str] = []
    for finding in findings:
        finding_lower = finding.lower()
        # Match if any keyword appears in the finding
        if any(kw in finding_lower for kw in keywords):
            matched.append(finding)

    return matched[:5]  # Limit to 5 findings per question


def _extract_keywords(text: str) -> List[str]:
    """Extract significant keywords from text.

    Removes common stopwords and returns meaningful words that can be
    used for finding matching.

    Args:
        text: Input text string.

    Returns:
        List of keyword strings (max 5, all lowercase).
    """
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "all",
        "any",
        "each",
        "how",
        "what",
        "which",
        "that",
        "this",
        "these",
        "those",
        "document",
        "list",
        "analyze",
        "evaluate",
        "identify",
        "assess",
        "catalog",
        "are",
        "is",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
    }

    # Extract words 3+ characters
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in stopwords][:5]


def _get_objective_confidence(
    objective_id: str,
    per_objective_scores: List[Dict[str, Any]],
) -> Optional[float]:
    """Extract confidence score for an objective from reflection scores.

    Searches the per_objective_scores list for a matching objective_id
    and returns its score, clamped to [0.0, 1.0].

    Args:
        objective_id: Objective identifier (e.g., "OBJ-1").
        per_objective_scores: List of score entries with objective_id and score.

    Returns:
        Float confidence in [0.0, 1.0], or None if not found.
    """
    for score_entry in per_objective_scores:
        if score_entry.get("objective_id") == objective_id:
            score = score_entry.get("score")
            if isinstance(score, (int, float)):
                return max(0.0, min(1.0, float(score)))
    return None


__all__ = [
    "KeyQuestion",
    "generate_key_questions",
]
