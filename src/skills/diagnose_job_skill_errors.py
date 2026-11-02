
import json
import re

import pandas as pd

from src.skills.skill_extractor import SkillExtractor
from src.skills.skill_taxonomy import normalize_skill


def load_skills(value):
    """Convert JSON skill annotations into canonical skill names."""
    if pd.isna(value):
        return set()

    try:
        skills = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return set()

    if not isinstance(skills, list):
        return set()

    result = set()

    for skill in skills:
        if isinstance(skill, str):
            result.add(normalize_skill(skill))

    return result


def get_context(text, keyword, window=100):
    """Show text surrounding a matched keyword."""
    if not isinstance(text, str):
        return "No description available."

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    match = pattern.search(text)

    if not match:
        return text[:250] + "..."

    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)

    return "..." + text[start:end] + "..."


def main():
    file_path = "data/processed/jobs_cleaned.csv"

    print("Loading job dataset...")

    df = pd.read_csv(file_path)
    extractor = SkillExtractor()

    missed_skills = [
        "Data Analysis",
        "Project Management",
        "Negotiation",
        "Customer Relationship Management",
    ]

    search_terms = {
        "Data Analysis": ["data", "analysis", "analytical"],
        "Project Management": ["project", "management", "managed"],
        "Negotiation": ["negotiation", "negotiating", "negotiated"],
        "Customer Relationship Management": [
            "customer", "client", "relationship"
        ],
    }

    print("\n========== FALSE NEGATIVE EXAMPLES ==========")

    for target_skill in missed_skills:
        print(f"\n\nTARGET SKILL: {target_skill}")
        shown = 0

        for _, row in df.iterrows():
            annotations = load_skills(row["clean_skills"])

            if target_skill not in annotations:
                continue

            description = row["cleaned_description"]
            predictions = extractor.extract_skills(description)
            predicted_names = {item["skill"] for item in predictions}

            if target_skill in predicted_names:
                continue

            print("\nJob ID:", row["job_id"])
            print("Title:", row["title"])

            terms = search_terms[target_skill]
            context_found = False

            for term in terms:
                context = get_context(description, term)

                if context != description[:250] + "...":
                    print(f"Context for '{term}': {context}")
                    context_found = True
                    break

            if not context_found:
                print("Description:", description[:300])

            shown += 1

            if shown == 3:
                break

        if shown == 0:
            print("No examples found.")

    print("\n\n========== UNANNOTATED PREDICTIONS ==========")

    shown = 0

    for _, row in df.iterrows():
        annotations = load_skills(row["clean_skills"])
        description = row["cleaned_description"]

        predictions = extractor.extract_skills(description)

        for prediction in predictions:
            skill = prediction["skill"]

            if skill not in annotations:
                print("\nJob ID:", row["job_id"])
                print("Title:", row["title"])
                print("Detected skill:", skill)
                print("Matched alias:", prediction["matched_alias"])

                context = get_context(
                    description,
                    prediction["matched_alias"]
                )
                print("Context:", context)

                shown += 1

                if shown == 10:
                    break

        if shown == 10:
            break

    print("\nDiagnosis completed.")


if __name__ == "__main__":
    main()
