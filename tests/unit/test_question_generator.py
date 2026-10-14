"""Unit tests for TraceLab KeyQuestion generation from mission objectives."""

from __future__ import annotations

import pytest

from deepsearch.agent.state import AgentState
from deepsearch.report.questions import (
    KeyQuestion,
    _extract_keywords,
    _format_question,
    _get_objective_confidence,
    _match_findings_to_objective,
    generate_key_questions,
)


class TestKeyQuestionModel:
    """Tests for the KeyQuestion Pydantic model."""

    def test_keyquestion_required_fields(self) -> None:
        """KeyQuestion should require question field."""
        kq = KeyQuestion(question="What is the answer?")
        assert kq.question == "What is the answer?"
        assert kq.status == "open"  # default

    def test_keyquestion_optional_fields_default(self) -> None:
        """Optional fields should default to None or 'open'."""
        kq = KeyQuestion(question="Test question?")
        assert kq.status == "open"
        assert kq.answer is None
        assert kq.confidence is None
        assert kq.owner is None

    def test_keyquestion_with_all_fields(self) -> None:
        """KeyQuestion should accept all fields."""
        kq = KeyQuestion(
            question="What is X?",
            status="answered",
            answer="X is a variable.",
            confidence=0.95,
            owner="researcher",
        )
        assert kq.question == "What is X?"
        assert kq.status == "answered"
        assert kq.answer == "X is a variable."
        assert kq.confidence == 0.95
        assert kq.owner == "researcher"

    def test_keyquestion_status_literal(self) -> None:
        """Status should only accept valid literals."""
        # Valid statuses
        for status in ["open", "researching", "answered"]:
            kq = KeyQuestion(
                question="Test?",
                status=status,
                answer="Answer" if status == "answered" else None,
            )
            assert kq.status == status

    def test_keyquestion_confidence_validation(self) -> None:
        """Confidence should be between 0.0 and 1.0."""
        # Valid confidence
        kq = KeyQuestion(question="Test?", confidence=0.5)
        assert kq.confidence == 0.5

        # Boundary values
        kq_min = KeyQuestion(question="Test?", confidence=0.0)
        assert kq_min.confidence == 0.0

        kq_max = KeyQuestion(question="Test?", confidence=1.0)
        assert kq_max.confidence == 1.0

    def test_keyquestion_confidence_out_of_range(self) -> None:
        """Confidence outside 0-1 range should raise validation error."""
        with pytest.raises(ValueError):
            KeyQuestion(question="Test?", confidence=1.5)

        with pytest.raises(ValueError):
            KeyQuestion(question="Test?", confidence=-0.1)

    def test_keyquestion_to_dict(self) -> None:
        """to_dict should return dict without None values."""
        kq = KeyQuestion(
            question="What is X?",
            status="answered",
            answer="X is defined here.",
        )
        data = kq.to_dict()

        assert data["question"] == "What is X?"
        assert data["status"] == "answered"
        assert data["answer"] == "X is defined here."
        # None values should be excluded
        assert "confidence" not in data
        assert "owner" not in data


class TestGenerateKeyQuestions:
    """Tests for generate_key_questions function."""

    def test_generate_questions_normal(self) -> None:
        """Normal case: 3 objectives should produce 3 KeyQuestion objects."""
        state: AgentState = {
            "mission_id": "TEST-001",
            "mission_objectives": [
                "Document the API endpoints",
                "Analyze performance bottlenecks",
                "What are the security considerations?",
            ],
            "findings": [
                "The API has 5 main endpoints for user management.",
                "Performance testing shows 200ms average response time.",
                "Security review identified 2 potential vulnerabilities.",
                "API documentation is available at /docs.",
            ],
            "reflection_scores": {
                "overall_coverage_score": 0.8,
                "per_objective_scores": [
                    {"objective_id": "OBJ-1", "score": 0.85, "reasoning": "Well covered"},
                    {"objective_id": "OBJ-2", "score": 0.75, "reasoning": "Partially covered"},
                    {"objective_id": "OBJ-3", "score": 0.7, "reasoning": "Some gaps"},
                ],
            },
        }

        questions = generate_key_questions(state)

        assert len(questions) == 3
        # Verify all are KeyQuestion objects
        for q in questions:
            assert isinstance(q, KeyQuestion)
            assert q.question.endswith("?")

    def test_generate_questions_empty_objectives(self) -> None:
        """Empty objectives should generate 1 default question."""
        state: AgentState = {
            "mission_id": "TEST-002",
            "mission_objectives": [],
            "findings": ["Some finding from research."],
            "reflection_scores": {"overall_coverage_score": 0.6},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        assert "TEST-002" in questions[0].question
        assert questions[0].status == "answered"

    def test_generate_questions_none_objectives(self) -> None:
        """None objectives should generate 1 default question."""
        state: AgentState = {
            "mission_id": "TEST-003",
            "mission_objectives": None,
            "findings": [],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        # S84 m04: no findings back the default question — it stays
        # honestly open (the pre-m04 branch fabricated an answered
        # placeholder with a dead-reflection confidence).
        assert questions[0].status == "open"
        assert questions[0].answer is None
        assert questions[0].confidence is None

    def test_generate_questions_missing_objectives_key(self) -> None:
        """Missing objectives key should generate default question."""
        state: AgentState = {"mission_id": "TEST-004"}

        questions = generate_key_questions(state)

        assert len(questions) == 1
        assert questions[0].question.endswith("?")

    def test_answered_only_when_real_findings_back_it(self) -> None:
        """S84 m04: 'answered' requires REAL findings — the pre-m04
        'at least one answered' promotion fabricated a placeholder answer
        with confidence=0.3 on every production mission (findings have no
        writer on the post-S64 path)."""
        # Case 1: Objectives with findings — genuinely answered.
        state1: AgentState = {
            "mission_id": "TEST-005",
            "mission_objectives": ["Research topic A", "Research topic B"],
            "findings": [
                "Finding about topic A with sufficient detail.",
                "Another finding about topic B with more content.",
            ],
            "reflection_scores": {},
        }
        questions1 = generate_key_questions(state1)
        answered1 = [q for q in questions1 if q.status == "answered"]
        assert len(answered1) >= 1

        # Case 2: Objectives with no findings — every question stays open;
        # no placeholder answer, no fabricated confidence.
        state2: AgentState = {
            "mission_id": "TEST-006",
            "mission_objectives": ["Research topic C"],
            "findings": [],
            "reflection_scores": {},
        }
        questions2 = generate_key_questions(state2)
        assert all(q.status == "open" for q in questions2)
        assert all(q.answer is None for q in questions2)
        assert (
            "Research conducted but specific findings pending synthesis."
            not in [q.answer for q in questions2]
        )

    def test_generate_questions_confidence_from_reflection_scores(self) -> None:
        """Confidence should be derived from per_objective_scores."""
        state: AgentState = {
            "mission_id": "TEST-007",
            "mission_objectives": ["First objective", "Second objective"],
            "findings": ["Finding 1 about first topic.", "Finding 2 about second topic."],
            "reflection_scores": {
                "per_objective_scores": [
                    {"objective_id": "OBJ-1", "score": 0.9},
                    {"objective_id": "OBJ-2", "score": 0.7},
                ],
            },
        }

        questions = generate_key_questions(state)

        assert questions[0].confidence == 0.9
        assert questions[1].confidence == 0.7

    def test_generate_questions_prefers_required_contract_objectives(self) -> None:
        """TraceLab questions should ignore raw non-research mission-objective noise."""
        state: AgentState = {
            "mission_id": "FORCE-SMOKE-11",
            "mission_objectives": [
                "Explain when to use each backend in practice",
                "Capture one concrete operational gotcha for each backend",
                "Do not include RedisSaver",
                "Include a comparison table with columns: Backend, Use When, Avoid When",
            ],
            "mission_contract": {
                "objectives": [
                    {
                        "id": "objective-0",
                        "text": "Compare checkpoint backends",
                        "required": False,
                    },
                    {
                        "id": "objective-1",
                        "text": "Explain when to use each backend in practice",
                        "required": True,
                    },
                    {
                        "id": "objective-2",
                        "text": "Capture one concrete operational gotcha for each backend",
                        "required": True,
                    },
                ]
            },
            "findings": [
                "MemorySaver is suitable for ephemeral local testing with simple workflows.",
                "PostgresSaver adds operational overhead but supports durable multi-worker state.",
            ],
            "reflection_scores": {
                "per_objective_scores": [
                    {"objective_id": "OBJ-1", "score": 0.9},
                    {"objective_id": "OBJ-2", "score": 0.8},
                ]
            },
        }

        questions = generate_key_questions(state)

        assert len(questions) == 2
        assert all("RedisSaver" not in question.question for question in questions)

    def test_generate_questions_no_reflection_scores(self) -> None:
        """Questions without reflection scores should have None confidence."""
        state: AgentState = {
            "mission_id": "TEST-008",
            "mission_objectives": ["Some objective"],
            "findings": ["Some finding"],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert questions[0].confidence is None

    def test_generate_questions_answer_non_empty_when_answered(self) -> None:
        """Answered questions must have non-empty answer."""
        state: AgentState = {
            "mission_id": "TEST-009",
            "mission_objectives": ["Research the topic thoroughly"],
            "findings": [
                "Comprehensive finding about the topic with lots of detail.",
                "Another finding with substantial information.",
            ],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        for q in questions:
            if q.status == "answered":
                assert q.answer is not None
                assert len(q.answer.strip()) > 0


class TestFormatQuestion:
    """Tests for _format_question helper function."""

    def test_format_question_already_question(self) -> None:
        """Question ending with ? should be unchanged."""
        assert _format_question("What is the API structure?") == "What is the API structure?"
        assert _format_question("How does it work?") == "How does it work?"

    def test_format_question_document_prefix(self) -> None:
        """'Document X' should become 'What are the X?'."""
        assert _format_question("Document the CLI commands") == "What are the CLI commands?"
        assert _format_question("Document authentication flows") == "What are the authentication flows?"

    def test_format_question_list_prefix(self) -> None:
        """'List X' should become 'What are the X?'."""
        assert _format_question("List all endpoints") == "What are the all endpoints?"

    def test_format_question_analyze_prefix(self) -> None:
        """'Analyze X' should become 'What is the analysis of X?'."""
        assert _format_question("Analyze performance") == "What is the analysis of performance?"
        assert _format_question("Evaluate security") == "What is the analysis of security?"

    def test_format_question_default_append(self) -> None:
        """Other statements should just get ? appended."""
        assert _format_question("Investigate the bug") == "Investigate the bug?"
        assert _format_question("Review the code") == "Review the code?"

    def test_format_question_empty_string(self) -> None:
        """Empty string should return default question."""
        assert _format_question("") == "What was researched?"
        assert _format_question("   ") == "What was researched?"


class TestMatchFindingsToObjective:
    """Tests for _match_findings_to_objective helper function."""

    def test_match_findings_keyword_matching(self) -> None:
        """Findings with matching keywords should be included."""
        objective = "Document the authentication system"
        findings = [
            "The authentication uses JWT tokens.",
            "Database schema includes user table.",
            "Authentication flow starts with login endpoint.",
            "Caching improves response times.",
        ]

        matched = _match_findings_to_objective(objective, findings)

        assert len(matched) == 2
        assert "authentication" in matched[0].lower()
        assert "authentication" in matched[1].lower()

    def test_match_findings_max_five(self) -> None:
        """Should return maximum 5 findings."""
        objective = "Research APIs"
        findings = [f"API finding number {i}" for i in range(10)]

        matched = _match_findings_to_objective(objective, findings)

        assert len(matched) <= 5

    def test_match_findings_empty_findings(self) -> None:
        """Empty findings list should return empty list."""
        matched = _match_findings_to_objective("Any objective", [])
        assert matched == []

    def test_match_findings_no_keywords(self) -> None:
        """Objective with no extractable keywords falls back to first 2."""
        objective = "the a an"  # All stopwords
        findings = ["Finding 1", "Finding 2", "Finding 3"]

        matched = _match_findings_to_objective(objective, findings)

        # Falls back to first 2 findings
        assert len(matched) == 2
        assert matched == ["Finding 1", "Finding 2"]

    def test_match_findings_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        objective = "Document API endpoints"
        findings = ["The API provides REST endpoints.", "Database uses SQL."]

        matched = _match_findings_to_objective(objective, findings)

        assert len(matched) == 1
        assert "API" in matched[0]


class TestExtractKeywords:
    """Tests for _extract_keywords helper function."""

    def test_extract_keywords_removes_stopwords(self) -> None:
        """Stopwords should be removed."""
        keywords = _extract_keywords("The document for the API with all features")
        assert "the" not in keywords
        assert "for" not in keywords
        assert "with" not in keywords
        assert "all" not in keywords

    def test_extract_keywords_lowercase(self) -> None:
        """Keywords should be lowercase."""
        keywords = _extract_keywords("Document the API Endpoints")
        for kw in keywords:
            assert kw == kw.lower()

    def test_extract_keywords_min_length(self) -> None:
        """Keywords shorter than 3 chars should be excluded."""
        keywords = _extract_keywords("an API to go on")
        assert "an" not in keywords
        assert "to" not in keywords
        assert "on" not in keywords

    def test_extract_keywords_max_five(self) -> None:
        """Should return maximum 5 keywords."""
        keywords = _extract_keywords(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        )
        assert len(keywords) <= 5

    def test_extract_keywords_empty_string(self) -> None:
        """Empty string should return empty list."""
        assert _extract_keywords("") == []


class TestGetObjectiveConfidence:
    """Tests for _get_objective_confidence helper function."""

    def test_get_confidence_found(self) -> None:
        """Should return score when objective_id matches."""
        scores = [
            {"objective_id": "OBJ-1", "score": 0.85},
            {"objective_id": "OBJ-2", "score": 0.72},
        ]

        assert _get_objective_confidence("OBJ-1", scores) == 0.85
        assert _get_objective_confidence("OBJ-2", scores) == 0.72

    def test_get_confidence_not_found(self) -> None:
        """Should return None when objective_id not found."""
        scores = [{"objective_id": "OBJ-1", "score": 0.85}]

        assert _get_objective_confidence("OBJ-99", scores) is None
        assert _get_objective_confidence("OBJ-2", scores) is None

    def test_get_confidence_empty_scores(self) -> None:
        """Should return None for empty scores list."""
        assert _get_objective_confidence("OBJ-1", []) is None

    def test_get_confidence_clamped(self) -> None:
        """Scores outside 0-1 should be clamped."""
        scores_high = [{"objective_id": "OBJ-1", "score": 1.5}]
        assert _get_objective_confidence("OBJ-1", scores_high) == 1.0

        scores_low = [{"objective_id": "OBJ-1", "score": -0.5}]
        assert _get_objective_confidence("OBJ-1", scores_low) == 0.0

    def test_get_confidence_invalid_score_type(self) -> None:
        """Non-numeric score should return None."""
        scores = [{"objective_id": "OBJ-1", "score": "high"}]
        assert _get_objective_confidence("OBJ-1", scores) is None


class TestSchemaValidation:
    """Tests for TraceLab schema compatibility."""

    def test_keyquestion_model_validates(self) -> None:
        """KeyQuestion should pass Pydantic validation."""
        kq = KeyQuestion(
            question="What is X?",
            status="answered",
            answer="X is defined.",
        )
        # Re-validate from dict
        validated = KeyQuestion.model_validate(kq.model_dump())
        assert validated.question == "What is X?"

    def test_generated_questions_validate(self) -> None:
        """Generated KeyQuestion objects should pass schema validation."""
        state: AgentState = {
            "mission_id": "SCHEMA-TEST",
            "mission_objectives": ["Test objective"],
            "findings": ["Test finding with enough content."],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        for q in questions:
            # Should not raise
            KeyQuestion.model_validate(q.model_dump())

    def test_question_ends_with_question_mark(self) -> None:
        """Every generated question should end with ?."""
        state: AgentState = {
            "mission_id": "QUESTION-MARK-TEST",
            "mission_objectives": [
                "Document the API",
                "Analyze performance",
                "What is the architecture?",
                "Review security",
            ],
            "findings": [],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        for q in questions:
            assert q.question.endswith("?"), f"Question does not end with ?: {q.question}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_objective_single_finding(self) -> None:
        """Single objective with single matching finding."""
        state: AgentState = {
            "mission_id": "EDGE-001",
            "mission_objectives": ["Research the topic"],
            "findings": ["Detailed finding about the research topic here."],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        assert questions[0].status == "answered"
        assert questions[0].answer is not None

    def test_many_objectives_no_findings(self) -> None:
        """S84 m04: many objectives with zero findings stay honestly open —
        no fabricated promotion of question[0]."""
        state: AgentState = {
            "mission_id": "EDGE-002",
            "mission_objectives": ["Obj 1", "Obj 2", "Obj 3", "Obj 4", "Obj 5"],
            "findings": [],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 5
        assert all(q.status == "open" for q in questions)
        assert all(q.answer is None for q in questions)

    def test_findings_match_multiple_objectives(self) -> None:
        """A finding can match multiple objectives."""
        state: AgentState = {
            "mission_id": "EDGE-003",
            "mission_objectives": ["Research authentication", "Document authentication"],
            "findings": ["The authentication system uses OAuth 2.0 tokens."],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        # Both should have the same finding matched
        for q in questions:
            if q.answer:
                assert "authentication" in q.answer.lower()

    def test_unicode_in_objectives_and_findings(self) -> None:
        """Should handle unicode characters correctly."""
        state: AgentState = {
            "mission_id": "EDGE-004",
            "mission_objectives": ["Rechercher l'API \u4e2d\u6587"],
            "findings": ["L'API utilise l'authentification OAuth."],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        # Should not raise any errors

    def test_very_long_objective(self) -> None:
        """Should handle very long objective strings."""
        long_objective = "Research " + " ".join(["word"] * 100)
        state: AgentState = {
            "mission_id": "EDGE-005",
            "mission_objectives": [long_objective],
            "findings": ["Finding with word in it."],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        assert questions[0].question.endswith("?")

    def test_very_long_finding(self) -> None:
        """Should handle very long finding strings."""
        long_finding = "Finding " + " ".join(["content"] * 1000)
        state: AgentState = {
            "mission_id": "EDGE-006",
            "mission_objectives": ["Document content"],
            "findings": [long_finding],
            "reflection_scores": {},
        }

        questions = generate_key_questions(state)

        assert len(questions) == 1
        # Answer should include the finding
        assert questions[0].answer is not None
