
from pathlib import Path

import pandas as pd

from src.models.hybrid_model import HybridMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"
MATCHES_FILE = RAW_DATA_DIR / "matches.csv"

TOP_K = 5
SAMPLE_SIZE = 100

WEIGHT_COMBINATIONS = [
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8)
]


def calculate_metrics(recommended_ids, relevant_ids):

    hits = sum(job_id in relevant_ids for job_id in recommended_ids)

    precision = hits / len(recommended_ids) if recommended_ids else 0.0

    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    reciprocal_rank = 0.0

    for rank, job_id in enumerate(recommended_ids, start=1):
        if job_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


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
    if "job_id" not in jobs.columns or "cleaned_description" not in jobs.columns:
        raise ValueError("Required job columns are missing.")

    if "candidate_id" not in candidates.columns or "cleaned_resume_text" not in candidates.columns:
        raise ValueError("Required candidate columns are missing.")

    if not {"candidate_id", "job_id"}.issubset(matches.columns):
        raise ValueError("Required match columns are missing.")

    # Reproducible candidate sample
    sample_candidates = candidates.sample(
        n=min(SAMPLE_SIZE, len(candidates)),
        random_state=42
    )

    job_descriptions = jobs["cleaned_description"].fillna("").tolist()
    job_ids = jobs["job_id"].tolist()

    # Cache the relevant job IDs for each candidate
    relevance = {}

    for _, candidate in sample_candidates.iterrows():

        candidate_id = candidate["candidate_id"]

        relevant_ids = set(
            matches.loc[
                matches["candidate_id"] == candidate_id,
                "job_id"
            ].tolist()
        )

        if relevant_ids:
            relevance[candidate_id] = relevant_ids

    if not relevance:
        raise ValueError("No sampled candidates have recorded positive matches.")

    print("\n========== HYBRID WEIGHT EXPERIMENT ==========")
    print(f"Candidates with recorded matches: {len(relevance)}")

    # Train one model per weight combination
    experiment_results = []

    for tfidf_weight, semantic_weight in WEIGHT_COMBINATIONS:

        print(
            f"\nTesting TF-IDF={tfidf_weight:.1f}, "
            f"Semantic={semantic_weight:.1f}"
        )

        model = HybridMatcher(
            tfidf_weight=tfidf_weight,
            semantic_weight=semantic_weight
        )

        model.fit(
            job_descriptions=job_descriptions,
            job_ids=job_ids
        )

        precision_scores = []
        recall_scores = []
        reciprocal_ranks = []

        for _, candidate in sample_candidates.iterrows():

            candidate_id = candidate["candidate_id"]

            if candidate_id not in relevance:
                continue

            results = model.rank_jobs(
                resume_text=candidate["cleaned_resume_text"],
                top_k=TOP_K
            )

            recommended_ids = [
                result["job_id"] for result in results
            ]

            precision, recall, mrr = calculate_metrics(
                recommended_ids,
                relevance[candidate_id]
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            reciprocal_ranks.append(mrr)

        experiment_results.append({
            "tfidf_weight": tfidf_weight,
            "semantic_weight": semantic_weight,
            "precision_at_5": sum(precision_scores) / len(precision_scores),
            "recall_at_5": sum(recall_scores) / len(recall_scores),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
        })

    # Display results
    results_df = pd.DataFrame(experiment_results)

    print("\n========== WEIGHT EXPERIMENT RESULTS ==========")
    print(results_df.to_string(index=False))

    # Identify best configuration by MRR
    best = results_df.loc[results_df["mrr"].idxmax()]

    print("\n========== BEST CONFIGURATION BY MRR ==========")
    print(f"TF-IDF weight: {best['tfidf_weight']:.1f}")
    print(f"Semantic weight: {best['semantic_weight']:.1f}")
    print(f"Precision@5: {best['precision_at_5']:.4f}")
    print(f"Recall@5: {best['recall_at_5']:.4f}")
    print(f"MRR: {best['mrr']:.4f}")

    print("\nWeight experiment completed successfully!")


if __name__ == "__main__":
    main()
