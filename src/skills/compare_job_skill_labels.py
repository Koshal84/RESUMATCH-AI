
import json
import pandas as pd

classifier_path = "data/processed/job_skill_classification_results.csv"
review_path = "data/processed/resumatch_ai_skill_review.csv"

classifier = pd.read_csv(classifier_path)
review = pd.read_csv(review_path)


def normalize_skill(skill):
    return str(skill).strip().lower()


# Convert classifier JSON columns into individual skill rows
classifier_rows = []

label_columns = {
    "required_skills": "Required",
    "preferred_skills": "Preferred",
    "unclassified_skills": "Unclassified"
}

for _, row in classifier.iterrows():
    for column, label in label_columns.items():
        skills = json.loads(row[column])

        for skill_data in skills:
            classifier_rows.append({
                "job_id": str(row["job_id"]),
                "skill": skill_data["skill"],
                "classifier_label": label
            })

classifier_df = pd.DataFrame(classifier_rows)

# Prepare AI-reviewed labels
review_df = review[[
    "job_id", "skill", "ai_review_label"
]].copy()

classifier_df["job_id"] = classifier_df["job_id"].astype(str)
review_df["job_id"] = review_df["job_id"].astype(str)

classifier_df["skill_key"] = classifier_df["skill"].apply(normalize_skill)
review_df["skill_key"] = review_df["skill"].apply(normalize_skill)

# Compare using job_id + skill
comparison = classifier_df.merge(
    review_df,
    on=["job_id", "skill_key"],
    how="inner",
    suffixes=("_classifier", "_review")
)

comparison["match"] = (
    comparison["classifier_label"] == comparison["ai_review_label"]
)

comparison["review_status"] = comparison["match"].map({
    True: "Agreement",
    False: "Disagreement"
})

# Save detailed comparison
output_path = "data/processed/skill_label_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nSKILL LABEL COMPARISON")
print("=====================")
print("Matched skill entries:", len(comparison))
print("Agreements:", comparison["match"].sum())
print("Disagreements:", (~comparison["match"]).sum())

print("\nDisagreement details:")
print(
    comparison.loc[
        ~comparison["match"],
        ["job_id", "skill_classifier", "classifier_label", "ai_review_label"]
    ].to_string(index=False)
)

print("\nResults saved to:", output_path)
