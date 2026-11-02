
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.models.tfidf_model import TFIDFMatcher
from src.models.semantic_model import SemanticMatcher
from src.models.hybrid_model import HybridMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"
MATCHES_FILE = RAW_DIR / "matches.csv"

TOP_K = 5
SAMPLE_SIZE = 100
RANDOM_STATE_TUNING = 42
RANDOM_STATE_HOLDOUT = 123


def evaluate_model(model, candidates, matches, model_name, job_ids):
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for _, candidate in candidates.iterrows():

        candidate_id = candidate["candidate_id"]

        actual_jobs = set(
            matches.loc[
                matches["candidate_id"] == candidate_id,
                "job_id"
            ]
        )

        if not actual_jobs:
            continue

        if model_name == "Hybrid 70:30":
            results = model.rank_jobs(
                resume_text=candidate["cleaned_resume_text"],
                top_k=TOP_K
            )
        else:
            results = model.rank_jobs(
                resume_text=candidate["cleaned_resume_text"],
                job_ids=job_ids,
                top_k=TOP_K
            )

        predicted_jobs = [
            result["job_id"] for result in results
        ]

        relevant_count = len(
            set(predicted_jobs) & actual_jobs
        )

        precision = relevant_count / TOP_K
        recall = relevant_count / len(actual_jobs)

        precisions.append(precision)
        recalls.append(recall)

        reciprocal_rank = 0.0

        for rank, job_id in enumerate(predicted_jobs, start=1):
            if job_id in actual_jobs:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return {
        "model": model_name,
        "evaluated_candidates": len(precisions),
        "precision@5": sum(precisions) / len(precisions),
        "recall@5": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }


def main():

    print("Loading datasets...")

    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)
    matches = pd.read_csv(MATCHES_FILE)

    job_ids = jobs["job_id"].tolist()
    job_descriptions = jobs["cleaned_description"].fillna("").tolist()

    # Reconstruct the original tuning sample
    _, tuning_sample = train_test_split(
        candidates,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE_TUNING
    )

    tuning_ids = set(tuning_sample["candidate_id"])

    # Keep only candidates not used in tuning
    remaining_candidates = candidates[
        ~candidates["candidate_id"].isin(tuning_ids)
    ]

    # Create separate holdout sample
    holdout_candidates = remaining_candidates.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE_HOLDOUT
    )

    holdout_ids = set(holdout_candidates["candidate_id"])

    print("\n========== HOLDOUT DATASET ==========")
    print("Tuning candidates:", len(tuning_ids))
    print("Holdout candidates:", len(holdout_ids))
    print("Overlapping candidates:", len(tuning_ids & holdout_ids))

    # Train TF-IDF model
    print("\nTraining TF-IDF model...")

    tfidf_model = TFIDFMatcher()
    tfidf_model.fit(job_descriptions)

    # Train Semantic model
    print("\nTraining semantic model...")

    semantic_model = SemanticMatcher()
    semantic_model.fit(job_descriptions)

    # Train Hybrid model with 70:30 weights
    print("\nTraining hybrid model (70:30)...")

    hybrid_model = HybridMatcher(
        tfidf_weight=0.7,
        semantic_weight=0.3
    )
    hybrid_model.fit(job_descriptions, job_ids)

    # Evaluate all models on the same holdout candidates
    print("\nEvaluating models...")

    results = []

    results.append(
        evaluate_model(
            tfidf_model,
            holdout_candidates,
            matches,
            "TF-IDF",
            job_ids
        )
    )

    results.append(
        evaluate_model(
            semantic_model,
            holdout_candidates,
            matches,
            "Semantic",
            job_ids
        )
    )

    results.append(
        evaluate_model(
            hybrid_model,
            holdout_candidates,
            matches,
            "Hybrid 70:30",
            job_ids
        )
    )

    results_df = pd.DataFrame(results)

    print("\n========== HOLDOUT EVALUATION RESULTS ==========")
    print(results_df.to_string(index=False))

    output_file = DATA_DIR / "holdout_evaluation_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
