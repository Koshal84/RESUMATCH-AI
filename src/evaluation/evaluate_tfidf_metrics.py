
from pathlib import Path

import pandas as pd

from src.models.tfidf_model import TFIDFMatcher


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"
MATCHES_FILE = RAW_DATA_DIR / "matches.csv"

TOP_K = 5
SAMPLE_SIZE = 100


def precision_at_k(recommended_ids, relevant_ids):
    hits = sum(job_id in relevant_ids for job_id in recommended_ids)
    return hits / len(recommended_ids) if recommended_ids else 0.0


def recall_at_k(recommended_ids, relevant_ids):
    if not relevant_ids:
        return 0.0

    hits = sum(job_id in relevant_ids for job_id in recommended_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(recommended_ids, relevant_ids):
    for rank, job_id in enumerate(recommended_ids, start=1):
        if job_id in relevant_ids:
            return 1 / rank

    return 0.0


def main():

    # Check files
    for file_path in [JOBS_FILE, CANDIDATES_FILE, MATCHES_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load datasets
    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)
    matches = pd.read_csv(MATCHES_FILE)

    # Validate columns
    required_columns = {
        "jobs": ["job_id", "cleaned_description"],
        "candidates": ["candidate_id", "cleaned_resume_text"],
        "matches": ["candidate_id", "job_id", "label_strength"]
    }

    for column in required_columns["jobs"]:
        if column not in jobs.columns:
            raise ValueError(f"Missing job column: {column}")

    for column in required_columns["candidates"]:
        if column not in candidates.columns:
            raise ValueError(f"Missing candidate column: {column}")

    for column in required_columns["matches"]:
        if column not in matches.columns:
            raise ValueError(f"Missing match column: {column}")

    # Train TF-IDF model
    model = TFIDFMatcher()

    model.fit(
        jobs["cleaned_description"].fillna("").tolist()
    )

    # Select 10 candidates
    sample_candidates = candidates.sample(
    n=min(SAMPLE_SIZE, len(candidates)),
    random_state=42
)

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    print("\n========== TF-IDF METRICS EVALUATION ==========")

    for _, candidate in sample_candidates.iterrows():

        candidate_id = candidate["candidate_id"]
        resume_text = candidate["cleaned_resume_text"]

        # Get recorded positive matches
        relevant_ids = set(
            matches.loc[
                matches["candidate_id"] == candidate_id,
                "job_id"
            ].tolist()
        )

        if not relevant_ids:
            print(f"\nSkipping {candidate_id}: no recorded positive matches")
            continue

        # Get top 5 recommendations
        results = model.rank_jobs(
            resume_text=resume_text,
            job_ids=jobs["job_id"].tolist(),
            top_k=TOP_K
        )

        recommended_ids = [
            result["job_id"] for result in results
        ]

        # Calculate metrics
        precision = precision_at_k(recommended_ids, relevant_ids)
        recall = recall_at_k(recommended_ids, relevant_ids)
        mrr = mean_reciprocal_rank(recommended_ids, relevant_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(mrr)

        print(f"\nCandidate: {candidate_id}")
        print(f"Recorded positive jobs: {len(relevant_ids)}")
        print(f"Precision@{TOP_K}: {precision:.4f}")
        print(f"Recall@{TOP_K}: {recall:.4f}")
        print(f"Reciprocal Rank: {mrr:.4f}")

    # Calculate macro averages
    if not precision_scores:
        raise ValueError("No candidates with recorded positive matches.")

    print("\n========== FINAL MACRO AVERAGES ==========")
    print(f"Evaluated candidates: {len(precision_scores)}")
    print(f"Precision@{TOP_K}: {sum(precision_scores) / len(precision_scores):.4f}")
    print(f"Recall@{TOP_K}: {sum(recall_scores) / len(recall_scores):.4f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
