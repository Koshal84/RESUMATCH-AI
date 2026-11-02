
from pathlib import Path

import pandas as pd

from src.models.semantic_model import SemanticMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"


def main():
    # Check input files
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Jobs file not found: {JOBS_FILE}")

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CANDIDATES_FILE}"
        )

    # Load datasets
    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    # Validate required columns
    if "job_id" not in jobs.columns or "cleaned_description" not in jobs.columns:
        raise ValueError("Required job columns are missing.")

    if "candidate_id" not in candidates.columns or "cleaned_resume_text" not in candidates.columns:
        raise ValueError("Required candidate columns are missing.")

    # Select first candidate for testing
    candidate = candidates.iloc[0]

    # Initialize and train semantic model
    model = SemanticMatcher()

    model.fit(
        jobs["cleaned_description"].fillna("").tolist()
    )

    # Generate top 5 recommendations
    results = model.rank_jobs(
        resume_text=candidate["cleaned_resume_text"],
        job_ids=jobs["job_id"].tolist(),
        top_k=5
    )

    print("\n========== SEMANTIC MATCHING RESULTS ==========")
    print(f"Candidate ID: {candidate['candidate_id']}")

    for rank, result in enumerate(results, start=1):
        job_title = jobs.loc[
            jobs["job_id"] == result["job_id"],
            "title"
        ].iloc[0]

        print(f"\nRank: {rank}")
        print(f"Job Title: {job_title}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")


if __name__ == "__main__":
    main()
