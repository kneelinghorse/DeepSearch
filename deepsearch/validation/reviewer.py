"""Reviewer adjudication schema and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping

ReviewerEnvironment = Literal["local", "deployed"]
ReviewerVerdict = Literal["accept", "accept_with_minor_notes", "revise", "reject"]

REQUIRED_REVIEWER_CATEGORIES: tuple[str, ...] = (
    "contract_fit",
    "structural_integrity",
    "bibliography_quality",
    "corpus_grounding",
    "practical_usefulness",
)
VALID_REVIEWER_ENVIRONMENTS: tuple[str, ...] = ("local", "deployed")
VALID_REVIEWER_VERDICTS: tuple[str, ...] = (
    "accept",
    "accept_with_minor_notes",
    "revise",
    "reject",
)


@dataclass(slots=True)
class ReviewerAdjudication:
    """Human reviewer judgment attached to an acceptance artifact bundle."""

    contract_id: str
    run_id: str
    environment: ReviewerEnvironment
    reviewer: str
    rubric_scores: dict[str, int]
    per_category_rationale: dict[str, str]
    verdict: ReviewerVerdict
    adjudicated_at: str

    def validate(self) -> None:
        """Validate adjudication shape and rubric constraints."""

        if not self.contract_id.strip():
            raise ValueError("contract_id is required")
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.environment not in VALID_REVIEWER_ENVIRONMENTS:
            raise ValueError(
                "environment must be one of "
                f"{', '.join(VALID_REVIEWER_ENVIRONMENTS)}"
            )
        if not self.reviewer.strip():
            raise ValueError("reviewer is required")
        if self.verdict not in VALID_REVIEWER_VERDICTS:
            raise ValueError(
                "verdict must be one of "
                f"{', '.join(VALID_REVIEWER_VERDICTS)}"
            )

        self._validate_categories(
            payload=self.rubric_scores,
            label="rubric_scores",
        )
        self._validate_categories(
            payload=self.per_category_rationale,
            label="per_category_rationale",
        )

        for category, raw_score in self.rubric_scores.items():
            if isinstance(raw_score, bool) or not isinstance(raw_score, int):
                raise ValueError(f"{category} score must be an integer")
            if raw_score < 0 or raw_score > 3:
                raise ValueError(f"{category} score must be between 0 and 3")

        for category, rationale in self.per_category_rationale.items():
            if not str(rationale).strip():
                raise ValueError(f"{category} rationale must be non-empty")

        score_average = sum(self.rubric_scores.values()) / len(REQUIRED_REVIEWER_CATEGORIES)
        if self.verdict == "accept" and score_average < 2.5:
            raise ValueError("accept verdict requires rubric score average >= 2.5")

        try:
            datetime.fromisoformat(self.adjudicated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("adjudicated_at must be an ISO-8601 timestamp") from exc

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable reviewer payload."""

        return {
            "contract_id": self.contract_id,
            "run_id": self.run_id,
            "environment": self.environment,
            "reviewer": self.reviewer,
            "rubric_scores": dict(self.rubric_scores),
            "per_category_rationale": dict(self.per_category_rationale),
            "verdict": self.verdict,
            "adjudicated_at": self.adjudicated_at,
        }

    def write_to(self, artifact_dir: Path) -> Path:
        """Persist the adjudication under ``artifact_dir`` using the locked naming scheme."""

        self.validate()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{self.run_id}_reviewer.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewerAdjudication":
        """Construct and validate an adjudication from a mapping payload."""

        adjudication = cls(
            contract_id=str(payload.get("contract_id") or "").strip(),
            run_id=str(payload.get("run_id") or "").strip(),
            environment=str(payload.get("environment") or "").strip(),  # type: ignore[arg-type]
            reviewer=str(payload.get("reviewer") or "").strip(),
            rubric_scores={
                str(key): int(value)
                for key, value in dict(payload.get("rubric_scores") or {}).items()
            },
            per_category_rationale={
                str(key): str(value).strip()
                for key, value in dict(payload.get("per_category_rationale") or {}).items()
            },
            verdict=str(payload.get("verdict") or "").strip(),  # type: ignore[arg-type]
            adjudicated_at=str(payload.get("adjudicated_at") or "").strip(),
        )
        adjudication.validate()
        return adjudication

    @classmethod
    def read_from(cls, path: Path) -> "ReviewerAdjudication":
        """Load, validate, and return a reviewer adjudication from JSON."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("reviewer adjudication JSON must contain an object payload")
        return cls.from_dict(payload)

    @staticmethod
    def new_timestamp() -> str:
        """Return the canonical UTC timestamp string for reviewer artifacts."""

        return datetime.now(timezone.utc).isoformat()

    def _validate_categories(self, *, payload: Mapping[str, Any], label: str) -> None:
        keys = set(payload)
        required = set(REQUIRED_REVIEWER_CATEGORIES)
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        if missing or extra:
            problems: list[str] = []
            if missing:
                problems.append(f"missing {', '.join(missing)}")
            if extra:
                problems.append(f"unexpected {', '.join(extra)}")
            raise ValueError(f"{label} categories invalid: {'; '.join(problems)}")


__all__ = [
    "REQUIRED_REVIEWER_CATEGORIES",
    "ReviewerAdjudication",
    "ReviewerEnvironment",
    "ReviewerVerdict",
    "VALID_REVIEWER_ENVIRONMENTS",
    "VALID_REVIEWER_VERDICTS",
]
