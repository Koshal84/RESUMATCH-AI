
import pandas as pd

df = pd.read_csv("data/processed/candidates_cleaned.csv")

resume = str(df.iloc[0]["resume_text"])

print("CANDIDATE:", df.iloc[0]["candidate_id"])
print("\nRESUME TEXT:\n")
print(resume[:2500])

print("\n\nSECTION HEADINGS FOUND:\n")

import re

pattern = re.compile(
    r"\b(?:skills|education|experience|projects|"
    r"certifications|summary|objective|target position)\s*:",
    re.IGNORECASE
)

for match in pattern.finditer(resume):
    print(repr(match.group(0)), "at position", match.start())
