
import pandas as pd

comparison = pd.read_csv(
    "data/processed/skill_label_comparison.csv"
)

review = pd.read_csv(
    "data/processed/resumatch_ai_skill_review.csv"
)

disagreements = comparison[
    comparison["review_status"] == "Disagreement"
].copy()

details = disagreements.merge(
    review[["job_id", "skill", "description", "review_reason"]],
    left_on=["job_id", "skill_classifier"],
    right_on=["job_id", "skill"],
    how="left"
)

details = details[[
    "job_id",
    "skill_classifier",
    "classifier_label",
    "ai_review_label",
    "review_reason",
    "description"
]]

details.to_csv(
    "data/processed/skill_disagreement_analysis.csv",
    index=False
)

print("Disagreements analyzed:", len(details))
print(
    "Saved to: data/processed/skill_disagreement_analysis.csv"
)
