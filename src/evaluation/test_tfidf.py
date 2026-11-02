from pathlib import Path

import pandas as pd

from src.models.tfidf_model import TFIDFMatcher


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"


def main():
    # Check whether files exist
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Jobs file not found: {JOBS_FILE}")

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CANDIDATES_FILE}"
        )

    # Load datasets
    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    # Check required columns
    required_job_columns = ["job_id", "cleaned_description"]
    required_candidate_columns = [
        "candidate_id",
        "cleaned_resume_text"
    ]

    for column in required_job_columns:
        if column not in jobs.columns:
            raise ValueError(f"Missing job column: {column}")

    for column in required_candidate_columns:
        if column not in candidates.columns:
            raise ValueError(f"Missing candidate column: {column}")

    # Select one candidate for testing
    resume_text = candidates.iloc[0]["cleaned_resume_text"]

    # Train TF-IDF model
    model = TFIDFMatcher()
    model.fit(jobs["cleaned_description"].fillna("").tolist())

    # Rank jobs
    results = model.rank_jobs(
        resume_text=resume_text,
        job_ids=jobs["job_id"].tolist(),
        top_k=5
    )

    # Display results
    print("\n========== TF-IDF MATCHING RESULTS ==========")

    for rank, result in enumerate(results, start=1):
        job_title = jobs.loc[
            jobs["job_id"] == result["job_id"], "title"
        ].iloc[0]

        print(f"\nRank: {rank}")
        print(f"Job ID: {result['job_id']}")
        print(f"Job Title: {job_title}")
        print(
            f"Similarity Score: "
            f"{result['similarity_score']:.4f}"
        )

    print("\nTF-IDF test completed successfully!")

if __name__ == "__main__":
    main()