
class ExplanationEngine:
    """
    Generates evidence-based explanations for resume-job matches.

    Uses:
    - SkillExtractor
    - JobSkillClassifier
    - SkillGapAnalyzer
    - HybridMatcher
    """

    def __init__(
        self,
        skill_extractor,
        job_skill_classifier,
        skill_gap_analyzer,
        hybrid_matcher
    ):
        self.skill_extractor = skill_extractor
        self.job_skill_classifier = job_skill_classifier
        self.skill_gap_analyzer = skill_gap_analyzer
        self.hybrid_matcher = hybrid_matcher

    def _extract_skills(self, text):
        """Extract skills from resume or job description."""

        return self.skill_extractor.extract_skills(text)

    def _get_resume_evidence(self, extracted_skills):
        """Collect evidence sentences supporting resume skills."""

        evidence = []

        for item in extracted_skills:
            evidence.append({
                "skill": item["skill"],
                "category": item["category"],
                "matched_alias": item["matched_alias"],
                "evidence_sentence": item["evidence_sentence"],
                "start_position": item["start_position"],
                "end_position": item["end_position"]
            })

        return evidence

    def explain_match(
        self,
        resume_text,
        job_description,
        job_id
    ):
        """
        Explain why a resume matches a particular job.

        The HybridMatcher must already be fitted.
        """

        # Validate input
        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        if not isinstance(job_description, str) or not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        if self.hybrid_matcher.job_ids is None:
            raise ValueError(
                "HybridMatcher is not fitted. Call fit() first."
            )

        if job_id not in self.hybrid_matcher.job_ids:
            raise ValueError(
                f"Job ID '{job_id}' was not found in the HybridMatcher."
            )

        # Step 1: Extract resume skills
        candidate_skills = self._extract_skills(resume_text)

        # Step 2: Extract job description skills
        extracted_job_skills = self._extract_skills(job_description)

        # Step 3: Classify job skills
        classification = self.job_skill_classifier.classify(
            job_description,
            extracted_job_skills
        )

        required_skills = classification["required_skills"]
        preferred_skills = classification["preferred_skills"]

        # Step 4: Analyze skill gaps
        skill_analysis = self.skill_gap_analyzer.analyze(
            candidate_skills,
            required_skills,
            preferred_skills
        )

        # Step 5: Calculate hybrid matching scores
        ranked_jobs = self.hybrid_matcher.rank_jobs(
            resume_text,
            top_k=len(self.hybrid_matcher.job_ids)
        )

        matching_job = next(
            job for job in ranked_jobs
            if job["job_id"] == job_id
        )

        # Step 6: Prepare explanation
        explanation = {
            "job_id": job_id,

            "matching_scores": {
                "tfidf_score": matching_job["tfidf_score"],
                "semantic_score": matching_job["semantic_score"],
                "hybrid_score": matching_job["hybrid_score"]
            },

            "skill_analysis": skill_analysis,

            "job_skill_classification": {
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "unclassified_skills": classification[
                    "unclassified_skills"
                ],
                "classification_evidence": classification[
                    "classification_evidence"
                ],
                "counts": classification["counts"]
            },

            "resume_evidence": self._get_resume_evidence(
                candidate_skills
            ),

            "summary": {
                "required_match_percentage": skill_analysis[
                    "required_match_percentage"
                ],
                "preferred_match_percentage": skill_analysis[
                    "preferred_match_percentage"
                ],
                "overall_skill_match_percentage": skill_analysis[
                    "overall_match_percentage"
                ],
                "matched_required_skills": skill_analysis[
                    "matched_required_skills"
                ],
                "missing_required_skills": skill_analysis[
                    "missing_required_skills"
                ],
                "matched_preferred_skills": skill_analysis[
                    "matched_preferred_skills"
                ],
                "missing_preferred_skills": skill_analysis[
                    "missing_preferred_skills"
                ]
            }
        }

        return explanation
