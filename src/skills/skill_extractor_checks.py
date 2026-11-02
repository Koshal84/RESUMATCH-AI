
import pandas as pd

from src.skills.skill_extractor import SkillExtractor


def main():

    candidates = pd.read_csv(
        "data/processed/candidates_cleaned.csv"
    )

    jobs = pd.read_csv(
        "data/processed/jobs_cleaned.csv"
    )

    extractor = SkillExtractor()

    print("\n========== RESUME SKILL EXTRACTION ==========")

    for _, candidate in candidates.head(5).iterrows():

        extracted = extractor.extract_skills(
            candidate["cleaned_resume_text"]
        )

        print("\nCandidate ID:", candidate["candidate_id"])
        print("Extracted skills:", [
            item["skill"] for item in extracted
        ])

    print("\n========== JOB SKILL EXTRACTION ==========")

    for _, job in jobs.head(5).iterrows():

        extracted = extractor.extract_skills(
            job["cleaned_description"]
        )

        print("\nJob ID:", job["job_id"])
        print("Job Title:", job["title"])
        print("Extracted skills:", [
            item["skill"] for item in extracted
        ])


if __name__ == "__main__":
    main()
