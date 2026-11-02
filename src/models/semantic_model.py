
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    """
    Semantic resume-job matching using Sentence Transformers.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

        self.job_embeddings = None

    def fit(self, job_descriptions):
        """
        Generate embeddings for all job descriptions.
        """

        self.job_embeddings = self.model.encode(
            job_descriptions,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return self

    def rank_jobs(self, resume_text, job_ids, top_k=10):
        """
        Rank jobs based on semantic similarity with a resume.
        """

        if self.job_embeddings is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        resume_embedding = self.model.encode(
            [resume_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            resume_embedding,
            self.job_embeddings
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "job_id": job_ids[index],
                "similarity_score": float(scores[index])
            })

        return results
