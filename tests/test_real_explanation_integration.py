import pandas as pd

from src.models.hybrid_model import HybridMatcher
from src.skills.skill_extractor import SkillExtractor
from src.skills.job_skill_classifier import JobSkillClassifier
from src.skills.skill_gap_analyzer import SkillGapAnalyzer
from src.explainability.explanation_engine import ExplanationEngine


def main():
    print("\nLoading datasets...")

    jobs = pd.read_csv("data/processed/jobs_cleaned.csv")
    candidates = pd.read_csv("data/processed/candidates_cleaned.csv")

    # Use a small sample to keep CPU and memory usage manageable.
    jobs_sample = jobs.head(20).copy()
    candidate = candidates.iloc[0]

    job_ids = jobs_sample["job_id"].astype(str).tolist()
    job_descriptions = jobs_sample["job_text"].fillna("").astype(str).tolist()

    print(f"Jobs loaded for model: {len(job_descriptions)}")
    print(f"Candidate selected: {candidate['candidate_id']}")

    print("\nFitting the real hybrid model...")
    matcher = HybridMatcher()
    matcher.fit(job_descriptions, job_ids)

    print("\nCreating explanation engine...")
    engine = ExplanationEngine(
        skill_extractor=SkillExtractor(),
        job_skill_classifier=JobSkillClassifier(),
        skill_gap_analyzer=SkillGapAnalyzer(),
        hybrid_matcher=matcher
    )

    # Explain the match against the first job in the sample.
    selected_job = jobs_sample.iloc[0]

    print("\nGenerating explanation...")
    result = engine.explain_match(
        resume_text=str(candidate["resume_text"]),
        job_description=str(selected_job["description"]),
        job_id=str(selected_job["job_id"])
    )

    print("\n========== REAL MODEL EXPLANATION ==========")

    print("\nCandidate ID:", candidate["candidate_id"])
    print("Job ID:", result["job_id"])

    print("\nMatching Scores:")
    print(result["matching_scores"])

    print("\nSkill Summary:")
    print(result["summary"])

    print("\nResume Evidence:")
    for item in result["resume_evidence"][:5]:
        print(item)

    print("\nJob Classification:")
    print(result["job_skill_classification"]["counts"])

    print("\n========== TEST COMPLETED ==========")


if __name__ == "__main__":
    main()