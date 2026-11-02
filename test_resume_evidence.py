
import pandas as pd

from src.skills.skill_extractor import SkillExtractor

df = pd.read_csv("data/processed/candidates_cleaned.csv")

resume = str(df.iloc[0]["resume_text"])

extractor = SkillExtractor()
skills = extractor.extract_skills(resume)

print("\nCandidate:", df.iloc[0]["candidate_id"])
print("\nExtracted Skill Evidence:\n")

for item in skills:
    print(
        f"Skill: {item['skill']}\n"
        f"Evidence: {item['evidence_sentence']}\n"
        f"Positions: {item['start_position']} - {item['end_position']}\n"
        f"{'-' * 60}"
    )
