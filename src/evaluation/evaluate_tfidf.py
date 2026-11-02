from pathlib import Path

import pandas as pd

from src.models.tfidf_model import TFIDFMatcher


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"


def main():

    # Check files
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Jobs file not found: {JOBS_FILE}")

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CANDIDATES_FILE}"
        )

    # Load datasets
    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    # Validate columns
    if "cleaned_description" not in jobs.columns:
        raise ValueError("Missing cleaned_description column")

    if "cleaned_resume_text" not in candidates.columns:
        raise ValueError("Missing cleaned_resume_text column")

    # Train TF-IDF model
    model = TFIDFMatcher()

    model.fit(
        jobs["cleaned_description"].fillna("").tolist()
    )

    # Select first 10 candidates
    sample_candidates = candidates.head(10)

    print("\n========== MULTI-CANDIDATE EVALUATION ==========")

    for _, candidate in sample_candidates.iterrows():

        candidate_id = candidate["candidate_id"]
        resume_text = candidate["cleaned_resume_text"]

        results = model.rank_jobs(
            resume_text=resume_text,
            job_ids=jobs["job_id"].tolist(),
            top_k=5
        )

        print(f"\nCandidate ID: {candidate_id}")

        for rank, result in enumerate(results, start=1):

            job_title = jobs.loc[
                jobs["job_id"] == result["job_id"],
                "title"
            ].iloc[0]

            print(
                f"Rank {rank}: {job_title} | "
                f"Score: {result['similarity_score']:.4f}"
            )

    print("\nMulti-candidate evaluation completed!")


if __name__ == "__main__":
    main()