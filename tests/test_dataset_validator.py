import pandas as pd
import pytest

from src.data.validate_dataset import (
    validate_job_csv,
    validate_dataset,
    validate_relationships,
)


def test_valid_dataset_passes(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame({
        "job_id": ["1", "2"],
        "title": ["Data Analyst", "Data Engineer"],
        "description": ["SQL and dashboards", "Python and pipelines"],
    }).to_csv(path, index=False)

    report = validate_job_csv(path)
    assert report["rows"] == 2
    assert report["status"] == "passed"


def test_missing_required_column_fails(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame({"title": ["Analyst"], "description": ["SQL"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="job_id"):
        validate_job_csv(path)


def test_empty_dataset_fails(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame(columns=["job_id", "title", "description"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no rows"):
        validate_job_csv(path)


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_job_csv(tmp_path / "missing.csv")


def test_valid_candidates_dataset_passes(tmp_path):
    path = tmp_path / "candidates.csv"

    pd.DataFrame({
        "candidate_id": ["cand_001", "cand_002"],
        "resume_text": ["Python developer", "Data analyst"],
    }).to_csv(path, index=False)

    report = validate_dataset(path, "candidates")

    assert report["rows"] == 2
    assert report["status"] == "passed"


def test_valid_matches_dataset_passes(tmp_path):
    path = tmp_path / "matches.csv"

    pd.DataFrame({
        "candidate_id": ["cand_001", "cand_002"],
        "job_id": ["job_001", "job_002"],
        "label_strength": ["strong_positive", "weak_positive"],
    }).to_csv(path, index=False)

    report = validate_dataset(path, "matches")

    assert report["rows"] == 2
    assert report["status"] == "passed"


def test_duplicate_candidate_job_pairs_detected(tmp_path):
    path = tmp_path / "matches.csv"

    pd.DataFrame({
        "candidate_id": ["cand_001", "cand_001"],
        "job_id": ["job_001", "job_001"],
        "label_strength": ["strong_positive", "weak_positive"],
    }).to_csv(path, index=False)

    report = validate_dataset(path, "matches")

    assert report["duplicate_candidate_job_pairs"] == 1
    assert report["status"] == "passed_with_warnings"


def test_unknown_candidate_id_detected(tmp_path):
    jobs_path = tmp_path / "jobs.csv"
    candidates_path = tmp_path / "candidates.csv"
    matches_path = tmp_path / "matches.csv"

    pd.DataFrame({
        "job_id": ["job_001"]
    }).to_csv(jobs_path, index=False)

    pd.DataFrame({
        "candidate_id": ["cand_001"]
    }).to_csv(candidates_path, index=False)

    pd.DataFrame({
        "candidate_id": ["cand_999"],
        "job_id": ["job_001"],
        "label_strength": ["strong_positive"],
    }).to_csv(matches_path, index=False)

    report = validate_relationships(
        jobs_path, candidates_path, matches_path
    )

    assert report["unknown_candidate_ids"] == 1
    assert report["unknown_job_ids"] == 0
    assert report["status"] == "failed"


def test_unknown_job_id_detected(tmp_path):
    jobs_path = tmp_path / "jobs.csv"
    candidates_path = tmp_path / "candidates.csv"
    matches_path = tmp_path / "matches.csv"

    pd.DataFrame({
        "job_id": ["job_001"]
    }).to_csv(jobs_path, index=False)

    pd.DataFrame({
        "candidate_id": ["cand_001"]
    }).to_csv(candidates_path, index=False)

    pd.DataFrame({
        "candidate_id": ["cand_001"],
        "job_id": ["job_999"],
        "label_strength": ["strong_positive"],
    }).to_csv(matches_path, index=False)

    report = validate_relationships(
        jobs_path, candidates_path, matches_path
    )

    assert report["unknown_job_ids"] == 1
    assert report["unknown_candidate_ids"] == 0
    assert report["status"] == "failed"
