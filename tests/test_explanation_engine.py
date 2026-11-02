
import pytest

from src.skills.skill_extractor import SkillExtractor
from src.skills.job_skill_classifier import JobSkillClassifier
from src.skills.skill_gap_analyzer import SkillGapAnalyzer
from src.explainability.explanation_engine import ExplanationEngine


class FakeHybridMatcher:
    """Lightweight matcher for testing the explanation engine."""

    def __init__(self):
        self.job_ids = ["JOB001", "JOB002"]

    def rank_jobs(self, resume_text, top_k=10):
        return [
            {
                "job_id": "JOB001",
                "tfidf_score": 0.75,
                "semantic_score": 0.82,
                "hybrid_score": 0.785
            },
            {
                "job_id": "JOB002",
                "tfidf_score": 0.40,
                "semantic_score": 0.55,
                "hybrid_score": 0.475
            }
        ][:top_k]


@pytest.fixture
def explanation_engine():
    """Create an explanation engine with project components."""

    return ExplanationEngine(
        skill_extractor=SkillExtractor(),
        job_skill_classifier=JobSkillClassifier(),
        skill_gap_analyzer=SkillGapAnalyzer(),
        hybrid_matcher=FakeHybridMatcher()
    )


def test_explanation_engine_returns_complete_explanation(
    explanation_engine
):
    resume_text = """
    I have experience working with Python and SQL.
    I have used Excel for data analysis.
    """

    job_description = """
    Data Analyst

    Required Skills:
    Python is required.
    SQL is required.

    Preferred Skills:
    Tableau is preferred.
    """

    result = explanation_engine.explain_match(
        resume_text=resume_text,
        job_description=job_description,
        job_id="JOB001"
    )

    assert result["job_id"] == "JOB001"

    assert result["matching_scores"]["tfidf_score"] == 0.75
    assert result["matching_scores"]["semantic_score"] == 0.82
    assert result["matching_scores"]["hybrid_score"] == 0.785

    assert "Python" in result["skill_analysis"]["matched_required_skills"]
    assert "SQL" in result["skill_analysis"]["matched_required_skills"]

    assert "Tableau" in result["skill_analysis"]["missing_preferred_skills"]

    assert result["summary"]["required_match_percentage"] == 100.0

    assert len(result["resume_evidence"]) > 0

    assert len(
        result["job_skill_classification"]["classification_evidence"]
    ) > 0


def test_explanation_engine_rejects_empty_resume(
    explanation_engine
):
    with pytest.raises(ValueError, match="Resume text cannot be empty"):
        explanation_engine.explain_match(
            resume_text="",
            job_description="Python is required.",
            job_id="JOB001"
        )


def test_explanation_engine_rejects_empty_job_description(
    explanation_engine
):
    with pytest.raises(
        ValueError,
        match="Job description cannot be empty"
    ):
        explanation_engine.explain_match(
            resume_text="I know Python.",
            job_description="",
            job_id="JOB001"
        )


def test_explanation_engine_rejects_unknown_job_id(
    explanation_engine
):
    with pytest.raises(ValueError, match="was not found"):
        explanation_engine.explain_match(
            resume_text="I know Python and SQL.",
            job_description="Python is required.",
            job_id="UNKNOWN_JOB"
        )


def test_explanation_engine_rejects_unfitted_matcher():
    matcher = FakeHybridMatcher()
    matcher.job_ids = None

    engine = ExplanationEngine(
        skill_extractor=SkillExtractor(),
        job_skill_classifier=JobSkillClassifier(),
        skill_gap_analyzer=SkillGapAnalyzer(),
        hybrid_matcher=matcher
    )

    with pytest.raises(ValueError, match="HybridMatcher is not fitted"):
        engine.explain_match(
            resume_text="I know Python.",
            job_description="Python is required.",
            job_id="JOB001"
        )
