
from pathlib import Path

import pandas as pd

from src.models.hybrid_model import HybridMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"


def main():

    # Check files
    for file_path in [JOBS_FILE, CANDIDATES_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load datasets
    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    # Select first candidate
    candidate = candidates.iloc[0]

    # Initialize hybrid model
    model = HybridMatcher(
        tfidf_weight=0.5,
        semantic_weight=0.5
    )

    # Train both models
    model.fit(
        job_descriptions=jobs["cleaned_description"].fillna("").tolist(),
        job_ids=jobs["job_id"].tolist()
    )

    # Get top 5 recommendations
    results = model.rank_jobs(
        resume_text=candidate["cleaned_resume_text"],
        top_k=5
    )

    print("\n========== HYBRID MATCHING RESULTS ==========")
    print(f"Candidate ID: {candidate['candidate_id']}")

    for rank, result in enumerate(results, start=1):

        job_title = jobs.loc[
            jobs["job_id"] == result["job_id"],
            "title"
        ].iloc[0]

        print(f"\nRank: {rank}")
        print(f"Job Title: {job_title}")
        print(f"TF-IDF Score: {result['tfidf_score']:.4f}")
        print(f"Semantic Score: {result['semantic_score']:.4f}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")


if __name__ == "__main__":
    main()
