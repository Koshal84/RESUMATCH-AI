
import json
from pathlib import Path

import pandas as pd

from src.skills.skill_extractor import SkillExtractor
from src.skills.skill_taxonomy import (
    SKILL_TAXONOMY,
    normalize_skill
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"
JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"


def load_annotated_skills(value):
    """Load only taxonomy-supported annotated skills."""

    if pd.isna(value):
        return set()

    try:
        skills = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return set()

    if not isinstance(skills, list):
        return set()

    normalized_skills = set()

    for skill in skills:
        if isinstance(skill, str):
            canonical_skill = normalize_skill(skill)

            if canonical_skill in SKILL_TAXONOMY:
                normalized_skills.add(canonical_skill)

    return normalized_skills


def evaluate_dataset(df, text_column, dataset_name):
    """Evaluate extraction using taxonomy-supported annotations."""

    extractor = SkillExtractor()

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    evaluated_records = 0

    skill_counts = {}

    for _, row in df.iterrows():

        actual_skills = load_annotated_skills(
            row["clean_skills"]
        )

        # Skip records without taxonomy-supported annotations.
        if not actual_skills:
            continue

        text = row[text_column]

        if not isinstance(text, str):
            text = ""

        predictions = extractor.extract_skills(text)

        predicted_skills = {
            item["skill"] for item in predictions
        }

        evaluated_records += 1

        tp = actual_skills & predicted_skills
        fp = predicted_skills - actual_skills
        fn = actual_skills - predicted_skills

        true_positives += len(tp)
        false_positives += len(fp)
        false_negatives += len(fn)

        for skill in tp:
            skill_counts.setdefault(
                skill,
                {"TP": 0, "FP": 0, "FN": 0}
            )
            skill_counts[skill]["TP"] += 1

        for skill in fp:
            skill_counts.setdefault(
                skill,
                {"TP": 0, "FP": 0, "FN": 0}
            )
            skill_counts[skill]["FP"] += 1

        for skill in fn:
            skill_counts.setdefault(
                skill,
                {"TP": 0, "FP": 0, "FN": 0}
            )
            skill_counts[skill]["FN"] += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    results = {
        "dataset": dataset_name,
        "evaluated_records": evaluated_records,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1_score, 6)
    }

    skill_report = []

    for skill, counts in skill_counts.items():

        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        skill_precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        skill_recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        skill_f1 = (
            2 * skill_precision * skill_recall /
            (skill_precision + skill_recall)
            if skill_precision + skill_recall > 0
            else 0
        )

        skill_report.append({
            "skill": skill,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(skill_precision, 6),
            "recall": round(skill_recall, 6),
            "f1_score": round(skill_f1, 6)
        })

    skill_report_df = pd.DataFrame(skill_report)

    if not skill_report_df.empty:
        skill_report_df = skill_report_df.sort_values(
            by=["FP", "FN"],
            ascending=False
        )

    return results, skill_report_df


def main():

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CANDIDATES_FILE}"
        )

    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            f"Jobs file not found: {JOBS_FILE}"
        )

    candidates = pd.read_csv(CANDIDATES_FILE)
    jobs = pd.read_csv(JOBS_FILE)

    candidate_required = {
        "clean_skills",
        "cleaned_resume_text"
    }

    job_required = {
        "clean_skills",
        "job_text"
    }

    if not candidate_required.issubset(candidates.columns):
        raise ValueError(
            "Required candidate columns are missing."
        )

    if not job_required.issubset(jobs.columns):
        raise ValueError(
            "Required job columns are missing. "
            "Run preprocessing again."
        )

    print("Evaluating candidate resumes...")

    candidate_results, candidate_skill_report = evaluate_dataset(
        candidates,
        "cleaned_resume_text",
        "Candidates"
    )

    print("Evaluating job titles and descriptions...")

    job_results, job_skill_report = evaluate_dataset(
        jobs,
        "job_text",
        "Jobs"
    )

    results = pd.DataFrame([
        candidate_results,
        job_results
    ])

    print("\n========== SKILL EXTRACTION EVALUATION ==========")
    print(results.to_string(index=False))

    results.to_csv(
        DATA_DIR / "skill_extraction_evaluation.csv",
        index=False
    )

    candidate_skill_report.to_csv(
        DATA_DIR / "candidate_skill_error_analysis.csv",
        index=False
    )

    job_skill_report.to_csv(
        DATA_DIR / "job_skill_error_analysis.csv",
        index=False
    )

    print("\nEvaluation files saved successfully:")
    print("1. skill_extraction_evaluation.csv")
    print("2. candidate_skill_error_analysis.csv")
    print("3. job_skill_error_analysis.csv")


if __name__ == "__main__":
    main()
