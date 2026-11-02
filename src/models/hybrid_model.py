
from sklearn.metrics.pairwise import cosine_similarity

from src.models.tfidf_model import TFIDFMatcher
from src.models.semantic_model import SemanticMatcher


class HybridMatcher:
    """
    Hybrid resume-job matching using TF-IDF and semantic similarity.
    """

    def __init__(
        self,
        tfidf_weight=0.5,
        semantic_weight=0.5
    ):
        if tfidf_weight < 0 or semantic_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if abs(tfidf_weight + semantic_weight - 1.0) > 1e-9:
            raise ValueError("Weights must add up to 1.")

        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight

        self.tfidf_matcher = TFIDFMatcher()
        self.semantic_matcher = SemanticMatcher()

        self.job_ids = None
        self.job_descriptions = None

    def fit(self, job_descriptions, job_ids):
        """
        Train both models on the same job descriptions.
        """

        if len(job_descriptions) != len(job_ids):
            raise ValueError(
                "Job descriptions and job IDs must have equal lengths."
            )

        if not job_descriptions:
            raise ValueError("Job descriptions cannot be empty.")

        self.job_descriptions = job_descriptions
        self.job_ids = job_ids

        self.tfidf_matcher.fit(job_descriptions)
        self.semantic_matcher.fit(job_descriptions)

        return self

    def rank_jobs(self, resume_text, top_k=10):
        """
        Rank jobs using a weighted combination of both models.
        """

        if self.job_ids is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Generate TF-IDF scores for every job
        resume_vector = self.tfidf_matcher.vectorizer.transform(
            [resume_text]
        )

        tfidf_scores = cosine_similarity(
            resume_vector,
            self.tfidf_matcher.job_vectors
        ).flatten()

        # Generate semantic scores for every job
        resume_embedding = self.semantic_matcher.model.encode(
            [resume_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_scores = cosine_similarity(
            resume_embedding,
            self.semantic_matcher.job_embeddings
        ).flatten()

        # Combine both scores
        hybrid_scores = (
            self.tfidf_weight * tfidf_scores
            + self.semantic_weight * semantic_scores
        )

        ranked_indices = hybrid_scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "job_id": self.job_ids[index],
                "tfidf_score": float(tfidf_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "hybrid_score": float(hybrid_scores[index])
            })

        return results
