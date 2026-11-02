
import pandas as pd

from src.models.hybrid_model import HybridMatcher


class JobMatchingService:

    def __init__(self, jobs_df):
        required_columns = {"job_id", "job_text"}

        missing = required_columns - set(jobs_df.columns)

        if missing:
            raise ValueError(
                f"Missing job dataset columns: {sorted(missing)}"
            )

        self.jobs_df = jobs_df.copy()

        self.jobs_df["job_text"] = (
            self.jobs_df["job_text"]
            .fillna("")
            .astype(str)
        )

        self.jobs_df = self.jobs_df[
            self.jobs_df["job_text"].str.strip() != ""
        ].reset_index(drop=True)

        if self.jobs_df.empty:
            raise ValueError("No usable job descriptions found.")

        self.matcher = HybridMatcher(
            tfidf_weight=0.7,
            semantic_weight=0.3
        )

        self.matcher.fit(
            job_descriptions=self.jobs_df["job_text"].tolist(),
            job_ids=self.jobs_df["job_id"].tolist()
        )

    def match_resume(self, resume_text, top_k=10):

        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        results = self.matcher.rank_jobs(
            resume_text=resume_text,
            top_k=top_k
        )

        results_df = pd.DataFrame(results)

        results_df = results_df.merge(
            self.jobs_df,
            on="job_id",
            how="left"
        )

        return results_df
