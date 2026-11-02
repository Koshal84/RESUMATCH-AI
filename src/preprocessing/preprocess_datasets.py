
from pathlib import Path
import pandas as pd

from src.preprocessing.clean_text import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

JOBS_FILE = DATA_DIR / "jobs_normalized.csv"
RAW_JOBS_FILE = RAW_DIR / "job.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_normalized.csv"


def preprocess_jobs():
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"File not found: {JOBS_FILE}")

    if not RAW_JOBS_FILE.exists():
        raise FileNotFoundError(f"File not found: {RAW_JOBS_FILE}")

    jobs = pd.read_csv(JOBS_FILE)
    raw_jobs = pd.read_csv(RAW_JOBS_FILE)

    required_normalized = {"job_id", "title", "description"}
    required_raw = {"job_id", "clean_skills"}

    if not required_normalized.issubset(jobs.columns):
        raise ValueError("Required columns missing from normalized jobs.")

    if not required_raw.issubset(raw_jobs.columns):
        raise ValueError("Required columns missing from original jobs.")

    if raw_jobs["job_id"].duplicated().any():
        raise ValueError("Duplicate job IDs found in original job data.")

    # Add original skill annotations to normalized jobs.
    jobs = jobs.merge(
        raw_jobs[["job_id", "clean_skills"]],
        on="job_id",
        how="left",
        validate="one_to_one"
    )

    if jobs["clean_skills"].isna().any():
        raise ValueError(
            "Some jobs could not be matched with original skills."
        )

    # Clean job titles and descriptions separately.
    jobs["cleaned_title"] = jobs["title"].apply(clean_text)

    jobs["cleaned_description"] = (
        jobs["description"].apply(clean_text)
    )

    # Combine title and description for skill extraction.
    jobs["job_text"] = (
        jobs["cleaned_title"] + " " + jobs["cleaned_description"]
    ).str.strip()

    output_file = DATA_DIR / "jobs_cleaned.csv"
    jobs.to_csv(output_file, index=False)

    print(f"Cleaned jobs saved: {output_file}")
    print(f"Total jobs processed: {len(jobs)}")
    print("Skills column added successfully.")
    print("Cleaned title and combined job text created successfully.")


def preprocess_candidates():
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"File not found: {CANDIDATES_FILE}")

    candidates = pd.read_csv(CANDIDATES_FILE)

    if "resume_text" not in candidates.columns:
        raise ValueError("Required column 'resume_text' is missing.")

    candidates["cleaned_resume_text"] = (
        candidates["resume_text"].apply(clean_text)
    )

    output_file = DATA_DIR / "candidates_cleaned.csv"
    candidates.to_csv(output_file, index=False)

    print(f"Cleaned candidates saved: {output_file}")
    print(f"Total candidates processed: {len(candidates)}")


if __name__ == "__main__":
    preprocess_jobs()
    preprocess_candidates()
