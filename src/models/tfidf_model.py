from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    """
    TF-IDF based resume-job matching model.
    """

    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True
        )

        self.job_vectors = None

    def fit(self, job_descriptions):
        """
        Learn vocabulary and IDF weights from job descriptions.
        """
        self.job_vectors = self.vectorizer.fit_transform(
            job_descriptions
        )

        return self

    def rank_jobs(self, resume_text, job_ids, top_k=10):
        """
        Rank jobs according to cosine similarity with a resume.
        """
        if self.job_vectors is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        resume_vector = self.vectorizer.transform([resume_text])

        scores = cosine_similarity(
            resume_vector,
            self.job_vectors
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "job_id": job_ids[index],
                "similarity_score": float(scores[index])
            })

        return results