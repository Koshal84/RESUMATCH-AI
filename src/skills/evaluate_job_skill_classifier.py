
import json
import pandas as pd

from src.skills.skill_extractor import SkillExtractor
from src.skills.job_skill_classifier import JobSkillClassifier


def main():

    input_path = "data/processed/jobs_cleaned.csv"

    output_path = (
        "data/processed/job_skill_classification_results.csv"
    )

    print("Loading job dataset...")

    jobs = pd.read_csv(input_path)

    extractor = SkillExtractor()
    classifier = JobSkillClassifier()

    results = []

    total_required = 0
    total_preferred = 0
    total_unclassified = 0

    jobs_with_required = 0
    jobs_with_preferred = 0
    jobs_with_unclassified = 0

    print("Evaluating job descriptions...")

    for index, row in jobs.iterrows():

        description = str(row["description"])

        extracted_skills = extractor.extract_skills(description)

        classification = classifier.classify(
            description,
            extracted_skills
        )

        required = classification["required_skills"]
        preferred = classification["preferred_skills"]
        unclassified = classification["unclassified_skills"]

        total_required += len(required)
        total_preferred += len(preferred)
        total_unclassified += len(unclassified)

        if required:
            jobs_with_required += 1

        if preferred:
            jobs_with_preferred += 1

        if unclassified:
            jobs_with_unclassified += 1

        results.append({

            "job_id": row["job_id"],
            "title": row["title"],

            "required_skills": json.dumps(required),
            "preferred_skills": json.dumps(preferred),
            "unclassified_skills": json.dumps(unclassified),

            "required_count": len(required),
            "preferred_count": len(preferred),
            "unclassified_count": len(unclassified),

            "classification_evidence": json.dumps(
                classification["classification_evidence"]
            )

        })

        if (index + 1) % 200 == 0:
            print(f"Processed {index + 1}/{len(jobs)} jobs")

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 45)
    print("JOB SKILL CLASSIFICATION EVALUATION")
    print("=" * 45)

    print("Total jobs:", len(jobs))

    print("\nJobs containing classified skills:")
    print("Required:", jobs_with_required)
    print("Preferred:", jobs_with_preferred)
    print("Unclassified:", jobs_with_unclassified)

    print("\nTotal skill classifications:")
    print("Required:", total_required)
    print("Preferred:", total_preferred)
    print("Unclassified:", total_unclassified)

    total_skills = (
        total_required + total_preferred + total_unclassified
    )

    print("\nTotal extracted skills:", total_skills)

    if total_skills:
        print(
            "Required percentage:",
            round(total_required / total_skills * 100, 2)
        )

        print(
            "Preferred percentage:",
            round(total_preferred / total_skills * 100, 2)
        )

        print(
            "Unclassified percentage:",
            round(total_unclassified / total_skills * 100, 2)
        )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
