
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"

# Exploratory threshold, not a universal quality standard
SHORT_TEXT_THRESHOLD = 100


def analyze_text_column(df, column, dataset_name):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {dataset_name}")

    text = df[column].fillna("").astype(str).str.strip()

    char_lengths = text.str.len()
    word_counts = text.str.split().str.len()

    print(f"\n--- {dataset_name} TEXT ANALYSIS ---")
    print(f"Total records: {len(df)}")
    print(f"Empty texts: {(char_lengths == 0).sum()}")
    print(
        f"Short texts (below {SHORT_TEXT_THRESHOLD} characters): "
        f"{(char_lengths < SHORT_TEXT_THRESHOLD).sum()}"
    )
    print(f"Average characters: {char_lengths.mean():.2f}")
    print(f"Median characters: {char_lengths.median():.2f}")
    print(f"Minimum characters: {char_lengths.min()}")
    print(f"Maximum characters: {char_lengths.max()}")
    print(f"Average words: {word_counts.mean():.2f}")


def main():
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Jobs file not found: {JOBS_FILE}")

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {CANDIDATES_FILE}"
        )

    jobs = pd.read_csv(JOBS_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    analyze_text_column(
        jobs,
        "cleaned_description",
        "JOBS"
    )

    analyze_text_column(
        candidates,
        "cleaned_resume_text",
        "CANDIDATES"
    )


if __name__ == "__main__":
    main()
