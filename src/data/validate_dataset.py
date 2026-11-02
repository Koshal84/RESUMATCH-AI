"""Validate RESUMATCH AI datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


SCHEMAS = {
    "jobs": {
        "required": {"job_id", "title", "description"},
        "unique_id": "job_id",
    },
    "candidates": {
        "required": {"candidate_id", "resume_text"},
        "unique_id": "candidate_id",
    },
    "matches": {
        "required": {"candidate_id", "job_id", "label_strength"},
        "unique_id": None,
    },
}


def load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Expected a file: {file_path}")

    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    return frame


def validate_dataset(path: str | Path, dataset_type: str) -> dict:
    if dataset_type not in SCHEMAS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    frame = load_csv(path)
    schema = SCHEMAS[dataset_type]

    missing = sorted(schema["required"] - set(frame.columns))

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    required = sorted(schema["required"])

    null_counts = {
        col: int(frame[col].isna().sum())
        for col in required
    }

    blank_counts = {
        col: int(
            frame[col].fillna("").astype(str).str.strip().eq("").sum()
        )
        for col in required
    }

    duplicate_ids = 0
    unique_id = schema["unique_id"]

    if unique_id:
        duplicate_ids = int(frame[unique_id].duplicated().sum())

    duplicate_pairs = 0

    if dataset_type == "matches":
        duplicate_pairs = int(
            frame.duplicated(subset=["candidate_id", "job_id"]).sum()
        )

    warnings = (
        any(null_counts.values())
        or any(blank_counts.values())
        or duplicate_ids > 0
        or duplicate_pairs > 0
    )

    return {
        "dataset_type": dataset_type,
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_counts": null_counts,
        "blank_counts": blank_counts,
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_job_pairs": duplicate_pairs,
        "status": "passed_with_warnings" if warnings else "passed",
    }


def validate_job_csv(path: str | Path) -> dict:
    """Backward-compatible job CSV validator."""
    return validate_dataset(path, "jobs")


def validate_relationships(
    jobs_path: str | Path,
    candidates_path: str | Path,
    matches_path: str | Path,
) -> dict:
    jobs = load_csv(jobs_path)
    candidates = load_csv(candidates_path)
    matches = load_csv(matches_path)

    for column in ["job_id"]:
        if column not in jobs.columns:
            raise ValueError(f"Jobs dataset missing {column}")

    if "candidate_id" not in candidates.columns:
        raise ValueError("Candidates dataset missing candidate_id")

    for column in ["candidate_id", "job_id"]:
        if column not in matches.columns:
            raise ValueError(f"Matches dataset missing {column}")

    known_jobs = set(jobs["job_id"].dropna().astype(str))
    known_candidates = set(
        candidates["candidate_id"].dropna().astype(str)
    )

    match_jobs = matches["job_id"].dropna().astype(str)
    match_candidates = matches["candidate_id"].dropna().astype(str)

    unknown_jobs = sorted(set(match_jobs) - known_jobs)
    unknown_candidates = sorted(set(match_candidates) - known_candidates)

    return {
        "total_match_pairs": len(matches),
        "unknown_job_ids": len(unknown_jobs),
        "unknown_candidate_ids": len(unknown_candidates),
        "status": (
            "passed"
            if not unknown_jobs and not unknown_candidates
            else "failed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate RESUMATCH AI datasets."
    )

    parser.add_argument("--input")
    parser.add_argument(
        "--type",
        choices=["jobs", "candidates", "matches"],
    )

    parser.add_argument("--check-relationships", action="store_true")
    parser.add_argument("--jobs")
    parser.add_argument("--candidates")
    parser.add_argument("--matches")

    args = parser.parse_args()

    try:
        if args.check_relationships:
            if not all([args.jobs, args.candidates, args.matches]):
                raise ValueError(
                    "Provide --jobs, --candidates and --matches."
                )

            report = validate_relationships(
                args.jobs,
                args.candidates,
                args.matches,
            )

        else:
            if not args.input or not args.type:
                raise ValueError(
                    "Provide both --input and --type."
                )

            report = validate_dataset(args.input, args.type)

    except (FileNotFoundError, ValueError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nRESUMATCH AI DATA VALIDATION REPORT")

    for key, value in report.items():
        print(f"{key}: {value}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
