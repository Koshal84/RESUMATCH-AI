import time
import re
import pandas as pd

from src.skills.skill_extractor import SkillExtractor
from src.skills.skill_taxonomy import SKILL_TAXONOMY


def build_legacy_patterns():
    """Build and compile legacy regex patterns once."""

    patterns = []

    for canonical_name, details in SKILL_TAXONOMY.items():

        aliases = [canonical_name] + details["aliases"]

        for alias in aliases:

            normalized_alias = alias.strip().lower()

            if (
                len(normalized_alias) < 3
                and normalized_alias not in {"c++", "c#"}
            ):
                continue

            pattern = re.compile(
                r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)",
                re.IGNORECASE
            )

            patterns.append((canonical_name, pattern))

    return patterns


def legacy_extract(text, patterns):
    """Extract skills using the legacy approach."""

    if not isinstance(text, str) or not text.strip():
        return set()

    found_skills = set()

    for canonical_name, pattern in patterns:

        if pattern.search(text):
            found_skills.add(canonical_name)

    return found_skills


def optimized_extract(text, extractor):
    """Extract skills using the optimized approach."""

    results = extractor.extract_skills(text)

    return {
        item["skill"]
        for item in results
    }


def main():

    print("Loading datasets...")

    candidates = pd.read_csv(
        "data/processed/candidates_cleaned.csv"
    )

    jobs = pd.read_csv(
        "data/processed/jobs_cleaned.csv"
    )

    resume_texts = (
        candidates["cleaned_resume_text"]
        .head(200)
        .fillna("")
        .tolist()
    )

    job_texts = (
        jobs["cleaned_description"]
        .head(200)
        .fillna("")
        .tolist()
    )

    documents = resume_texts + job_texts

    print("Total documents:", len(documents))

    print("\nPreparing extractors...")

    optimized_extractor = SkillExtractor()
    legacy_patterns = build_legacy_patterns()

    # -------------------------------
    # Legacy benchmark
    # -------------------------------

    print("\nRunning legacy extractor...")

    start = time.perf_counter()

    legacy_results = [
        legacy_extract(text, legacy_patterns)
        for text in documents
    ]

    legacy_time = time.perf_counter() - start

    # -------------------------------
    # Optimized benchmark
    # -------------------------------

    print("Running optimized extractor...")

    start = time.perf_counter()

    optimized_results = [
        optimized_extract(text, optimized_extractor)
        for text in documents
    ]

    optimized_time = time.perf_counter() - start

    # -------------------------------
    # Compare results
    # -------------------------------

    identical = sum(
        old == new
        for old, new in zip(legacy_results, optimized_results)
    )

    different = len(documents) - identical

    print("\n========== BENCHMARK RESULTS ==========")

    print(f"Legacy time: {legacy_time:.4f} seconds")
    print(f"Optimized time: {optimized_time:.4f} seconds")

    if optimized_time > 0:
        speedup = legacy_time / optimized_time
        print(f"Speedup: {speedup:.2f}x")

    print(f"Identical results: {identical}/{len(documents)}")
    print(f"Different results: {different}")

    agreement = (identical / len(documents)) * 100

    print(f"Result agreement: {agreement:.2f}%")

    # -------------------------------
    # Investigate differences
    # -------------------------------

    print("\n========== DIFFERENCE DETAILS ==========")

    difference_count = 0

    for index, (old, new) in enumerate(
        zip(legacy_results, optimized_results)
    ):

        if old != new:

            difference_count += 1

            text = documents[index]

            print(f"\nDocument index: {index}")

            if index < len(resume_texts):
                print("Document type: Candidate resume")
            else:
                print("Document type: Job description")

            print("Legacy only:", sorted(old - new))
            print("Optimized only:", sorted(new - old))

            # SQL-specific investigation
            if "SQL" in old:

                print("\nSQL occurrences in document:")

                matches = list(
                    re.finditer(r"sql", text, re.IGNORECASE)
                )

                if matches:

                    for match in matches:

                        start_context = max(
                            0,
                            match.start() - 80
                        )

                        end_context = min(
                            len(text),
                            match.end() + 80
                        )

                        print(
                            "Context:",
                            text[start_context:end_context]
                        )

                else:
                    print("No SQL occurrence found.")

                print(
                    "\nSQL alias mapping:",
                    optimized_extractor.alias_to_skill.get("sql")
                )

                print(
                    "Optimized SQL regex match:",
                    bool(
                        re.search(
                            r"(?<!\w)sql(?!\w)",
                            text,
                            re.IGNORECASE
                        )
                    )
                )

    if difference_count == 0:
        print("No differences found!")

    print("\n========== BENCHMARK COMPLETED ==========")


if __name__ == "__main__":
    main()