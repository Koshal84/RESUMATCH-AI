
from src.skills.skill_extractor import SkillExtractor


def test_extracts_python_and_sql():
    extractor = SkillExtractor()

    text = "Developed data pipelines using Python and SQL."

    results = extractor.extract_skills(text)
    skills = {item["skill"]: item for item in results}

    assert "Python" in skills
    assert "SQL" in skills


def test_preserves_existing_output_fields():
    extractor = SkillExtractor()

    results = extractor.extract_skills(
        "Experience working with Python."
    )

    python_skill = next(
        item for item in results
        if item["skill"] == "Python"
    )

    assert "skill" in python_skill
    assert "category" in python_skill
    assert "matched_alias" in python_skill


def test_includes_resume_evidence():
    extractor = SkillExtractor()

    text = "Developed data pipelines using Python and SQL."

    results = extractor.extract_skills(text)

    python_skill = next(
        item for item in results
        if item["skill"] == "Python"
    )

    assert python_skill["evidence_sentence"] == text
    assert python_skill["start_position"] >= 0
    assert python_skill["end_position"] > python_skill["start_position"]


def test_evidence_positions_match_skill_alias():
    extractor = SkillExtractor()

    text = "Built applications using Python."

    results = extractor.extract_skills(text)

    python_skill = next(
        item for item in results
        if item["skill"] == "Python"
    )

    extracted_text = text[
        python_skill["start_position"]:
        python_skill["end_position"]
    ]

    assert extracted_text.lower() == python_skill["matched_alias"].lower()


def test_empty_text_returns_empty_list():
    extractor = SkillExtractor()

    assert extractor.extract_skills("") == []
    assert extractor.extract_skills(None) == []
