
from src.knowledge_graph.skill_graph import SkillKnowledgeGraph


def test_add_skill():
    graph = SkillKnowledgeGraph()

    result = graph.add_skill("Python")

    assert result == "Python"
    assert "Python" in graph.nodes


def test_weighted_relationship():
    graph = SkillKnowledgeGraph()

    graph.add_related_skills(
        "Python", "SQL", weight=5, similarity=0.5
    )

    related = graph.get_related_skills("Python")

    assert related[0]["skill"] == "SQL"
    assert related[0]["weight"] == 5
    assert related[0]["jaccard_similarity"] == 0.5


def test_relationship_is_bidirectional():
    graph = SkillKnowledgeGraph()

    graph.add_related_skills("Python", "Pandas", weight=3)

    assert graph.get_related_skills("Python")[0]["skill"] == "Pandas"
    assert graph.get_related_skills("Pandas")[0]["skill"] == "Python"


def test_jaccard_calculation():
    result = SkillKnowledgeGraph.calculate_jaccard(100, 80, 60)

    assert result == 0.5


def test_jaccard_zero_union():
    result = SkillKnowledgeGraph.calculate_jaccard(0, 0, 0)

    assert result == 0.0


def test_minimum_weight_filter():
    graph = SkillKnowledgeGraph()

    graph.add_related_skills("Python", "SQL", weight=2, similarity=0.5)

    assert graph.get_related_skills("Python", min_weight=3) == []


def test_minimum_similarity_filter():
    graph = SkillKnowledgeGraph()

    graph.add_related_skills("Python", "SQL", weight=5, similarity=0.05)

    assert graph.get_related_skills(
        "Python", min_similarity=0.10
    ) == []


def test_category_skills():
    graph = SkillKnowledgeGraph()

    graph.add_skill("Python")

    assert "Python" in graph.get_category_skills(
        "Programming Languages"
    )


def test_graph_summary():
    graph = SkillKnowledgeGraph()

    graph.add_related_skills("Python", "SQL", weight=3)

    summary = graph.get_graph_summary()

    assert summary["skill_nodes"] == 2
    assert summary["unique_skill_relationships"] == 1


def test_unknown_skill():
    graph = SkillKnowledgeGraph()

    assert graph.get_related_skills("UnknownSkill") == []
