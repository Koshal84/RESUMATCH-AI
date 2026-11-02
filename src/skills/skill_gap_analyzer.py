
from src.skills.skill_taxonomy import (
    normalize_skill,
    get_skill_category
)


class SkillGapAnalyzer:

    def __init__(self):
        pass

    def _normalize_skills(self, skills):
        """
        Normalize skill names and remove duplicates.
        Accepts strings or SkillExtractor output dictionaries.
        """

        normalized = set()

        for item in skills or []:

            if isinstance(item, dict):
                skill = item.get("skill")
            else:
                skill = item

            if not isinstance(skill, str) or not skill.strip():
                continue

            canonical_skill = normalize_skill(skill)

            if canonical_skill:
                normalized.add(canonical_skill)

        return normalized

    def _calculate_percentage(self, matched, total):

        if total == 0:
            return 0.0

        return round((matched / total) * 100, 2)

    def _group_by_category(self, skills):

        grouped = {}

        for skill in sorted(skills):

            category = get_skill_category(skill)

            grouped.setdefault(category, []).append(skill)

        return grouped

    def analyze(
        self,
        candidate_skills,
        job_skills,
        preferred_skills=None
    ):
        """
        Compare candidate skills with job requirements.

        Parameters:
            candidate_skills: Skills extracted from resume.
            job_skills: Required job skills.
            preferred_skills: Optional preferred job skills.

        Returns:
            Required, preferred and overall skill gap analysis.
        """

        candidate_set = self._normalize_skills(candidate_skills)

        required_set = self._normalize_skills(job_skills)

        preferred_set = self._normalize_skills(preferred_skills)

        # A skill should not be counted twice.
        preferred_set -= required_set

        # Required skill comparison
        matched_required = candidate_set & required_set

        missing_required = required_set - candidate_set

        # Preferred skill comparison
        matched_preferred = candidate_set & preferred_set

        missing_preferred = preferred_set - candidate_set

        # All job skills
        all_job_skills = required_set | preferred_set

        matched_all = matched_required | matched_preferred

        # Additional candidate skills
        extra_skills = candidate_set - all_job_skills

        # Match percentages
        required_percentage = self._calculate_percentage(
            len(matched_required),
            len(required_set)
        )

        preferred_percentage = self._calculate_percentage(
            len(matched_preferred),
            len(preferred_set)
        )

        overall_percentage = self._calculate_percentage(
            len(matched_all),
            len(all_job_skills)
        )

        return {

            "candidate_skill_count": len(candidate_set),

            "required_skill_count": len(required_set),
            "preferred_skill_count": len(preferred_set),

            "matched_required_count": len(matched_required),
            "missing_required_count": len(missing_required),

            "matched_preferred_count": len(matched_preferred),
            "missing_preferred_count": len(missing_preferred),

            "extra_count": len(extra_skills),

            "required_match_percentage": required_percentage,
            "preferred_match_percentage": preferred_percentage,
            "overall_match_percentage": overall_percentage,

            "matched_required_skills": sorted(matched_required),
            "missing_required_skills": sorted(missing_required),

            "matched_preferred_skills": sorted(matched_preferred),
            "missing_preferred_skills": sorted(missing_preferred),

            "extra_skills": sorted(extra_skills),

            "category_wise_required_gaps": self._group_by_category(
                missing_required
            ),

            "category_wise_preferred_gaps": self._group_by_category(
                missing_preferred
            ),

            "category_wise_matches": self._group_by_category(
                matched_all
            )
        }
