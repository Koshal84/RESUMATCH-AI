
import pandas as pd

from src.models.hybrid_model import HybridMatcher
from src.skills.skill_extractor import SkillExtractor
from src.skills.job_skill_classifier import JobSkillClassifier
from src.skills.skill_gap_analyzer import SkillGapAnalyzer
from src.explainability.explanation_engine import ExplanationEngine


def main():

    print("\nLoading datasets...")

    jobs = pd.read_csv("data/processed/jobs_cleaned.csv")
    candidates = pd.read_csv(
        "data/processed/candidates_cleaned.csv"
    )

    # Small reproducible audit sample.
    jobs_sample = jobs.head(20).copy()
    candidates_sample = candidates.head(5).copy()

    job_ids = jobs_sample["job_id"].astype(str).tolist()

    job_descriptions = (
        jobs_sample["job_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("Jobs used for model:", len(jobs_sample))
    print("Candidates used for audit:", len(candidates_sample))

    print("\nFitting hybrid model...")

    matcher = HybridMatcher()
    matcher.fit(job_descriptions, job_ids)

    engine = ExplanationEngine(
        skill_extractor=SkillExtractor(),
        job_skill_classifier=JobSkillClassifier(),
        skill_gap_analyzer=SkillGapAnalyzer(),
        hybrid_matcher=matcher,
    )

    audit_results = []

    total_checks = 0
    passed_checks = 0

    print("\nStarting explainability audit...")

    for _, candidate in candidates_sample.iterrows():

        resume_text = str(candidate["resume_text"])
        candidate_id = str(candidate["candidate_id"])

        for _, job in jobs_sample.head(5).iterrows():

            job_id = str(job["job_id"])
            job_description = str(job["description"])

            total_checks += 1

            try:

                result = engine.explain_match(
                    resume_text=resume_text,
                    job_description=job_description,
                    job_id=job_id,
                )

                scores = result["matching_scores"]
                summary = result["summary"]
                skill_analysis = result["skill_analysis"]
                classification = result["job_skill_classification"]

                # Check 1: Matching scores are valid.
                scores_valid = all(
                    0 <= float(scores[key]) <= 1
                    for key in [
                        "tfidf_score",
                        "semantic_score",
                        "hybrid_score",
                    ]
                )

                # Check 2: Resume evidence positions are valid.
                evidence_valid = True

                for evidence in result["resume_evidence"]:

                    start = evidence["start_position"]
                    end = evidence["end_position"]
                    alias = evidence["matched_alias"]

                    extracted_text = resume_text[start:end]

                    if alias.casefold() not in extracted_text.casefold():
                        evidence_valid = False
                        break

                # Check 3: Classification counts agree with lists.
                classification_valid = (
                    classification["counts"]["required"]
                    == len(classification["required_skills"])
                    and classification["counts"]["preferred"]
                    == len(classification["preferred_skills"])
                    and classification["counts"]["unclassified"]
                    == len(classification["unclassified_skills"])
                )

                # Check 4: Skill gap counts agree with lists.
                gap_counts_valid = (
                    skill_analysis["matched_required_count"]
                    == len(skill_analysis["matched_required_skills"])
                    and skill_analysis["missing_required_count"]
                    == len(skill_analysis["missing_required_skills"])
                    and skill_analysis["matched_preferred_count"]
                    == len(skill_analysis["matched_preferred_skills"])
                    and skill_analysis["missing_preferred_count"]
                    == len(skill_analysis["missing_preferred_skills"])
                )

                # Check 5: Summary agrees with skill analysis.
                summary_valid = (
                    summary["required_match_percentage"]
                    == skill_analysis["required_match_percentage"]
                    and summary["preferred_match_percentage"]
                    == skill_analysis["preferred_match_percentage"]
                    and summary["overall_skill_match_percentage"]
                    == skill_analysis["overall_match_percentage"]
                )

                checks = [
                    scores_valid,
                    evidence_valid,
                    classification_valid,
                    gap_counts_valid,
                    summary_valid,
                ]

                checks_passed = sum(checks)
                passed_checks += checks_passed

                audit_results.append({
                    "candidate_id": candidate_id,
                    "job_id": job_id,
                    "tfidf_score": scores["tfidf_score"],
                    "semantic_score": scores["semantic_score"],
                    "hybrid_score": scores["hybrid_score"],
                    "resume_evidence_count": len(
                        result["resume_evidence"]
                    ),
                    "required_match_percentage":
                        summary["required_match_percentage"],
                    "preferred_match_percentage":
                        summary["preferred_match_percentage"],
                    "overall_skill_match_percentage":
                        summary["overall_skill_match_percentage"],
                    "scores_valid": scores_valid,
                    "evidence_valid": evidence_valid,
                    "classification_valid": classification_valid,
                    "gap_counts_valid": gap_counts_valid,
                    "summary_valid": summary_valid,
                    "checks_passed": checks_passed,
                    "total_checks": len(checks),
                    "status": (
                        "PASS" if all(checks) else "REVIEW"
                    ),
                })

                print(
                    f"{candidate_id} | {job_id} | "
                    f"{checks_passed}/5 checks passed"
                )

            except Exception as error:

                print(
                    f"ERROR: Candidate {candidate_id}, "
                    f"Job {job_id}: {error}"
                )

                audit_results.append({
                    "candidate_id": candidate_id,
                    "job_id": job_id,
                    "status": "ERROR",
                    "error": str(error),
                })

    # Save audit results.
    output_path = (
        "data/processed/explainability_audit_results.csv"
    )

    results_df = pd.DataFrame(audit_results)
    results_df.to_csv(output_path, index=False)

    print("\n========== EXPLAINABILITY AUDIT ==========")

    print("Total candidate-job pairs:", total_checks)
    print("Total checks:", total_checks * 5)
    print("Passed checks:", passed_checks)

    if total_checks > 0:
        print(
            "Check pass rate:",
            round(
                passed_checks / (total_checks * 5) * 100,
                2,
            ),
            "%",
        )

    print("\nPair status:")
    if "status" in results_df.columns:
        print(results_df["status"].value_counts().to_dict())

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
