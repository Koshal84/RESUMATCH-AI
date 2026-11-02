
from collections import defaultdict, Counter

from src.skills.skill_taxonomy import (
    normalize_skill,
    get_skill_category,
)


class SkillKnowledgeGraph:

    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(dict)

        self.skill_frequency = Counter()
        self.co_occurrence = Counter()

    def add_skill(self, skill):

        normalized = normalize_skill(skill)

        if not normalized:
            return None

        category = get_skill_category(normalized)

        self.nodes[normalized] = {
            "name": normalized,
            "type": "skill",
            "category": category,
        }

        category_node = f"category::{category}"

        self.nodes[category_node] = {
            "name": category,
            "type": "category",
        }

        self.edges[normalized][category_node] = {
            "relationship": "BELONGS_TO",
            "weight": 1,
        }

        return normalized

    def add_related_skills(
        self,
        skill_a,
        skill_b,
        weight=1,
        similarity=None,
    ):

        first = self.add_skill(skill_a)
        second = self.add_skill(skill_b)

        if not first or not second or first == second:
            return False

        weight = int(weight)

        if weight < 1:
            raise ValueError("Weight must be at least 1.")

        if similarity is None:
            similarity = 0.0

        if not 0 <= similarity <= 1:
            raise ValueError("Similarity must be between 0 and 1.")

        self.co_occurrence[(first, second)] += weight
        self.co_occurrence[(second, first)] += weight

        self.edges[first][second] = {
            "relationship": "RELATED_TO",
            "weight": self.co_occurrence[(first, second)],
            "jaccard_similarity": float(similarity),
        }

        self.edges[second][first] = {
            "relationship": "RELATED_TO",
            "weight": self.co_occurrence[(second, first)],
            "jaccard_similarity": float(similarity),
        }

        return True

    def set_skill_frequency(self, frequencies):

        self.skill_frequency.update(frequencies)

    @staticmethod
    def calculate_jaccard(
        skill_a_count,
        skill_b_count,
        co_occurrence_count,
    ):

        union_count = (
            skill_a_count
            + skill_b_count
            - co_occurrence_count
        )

        if union_count <= 0:
            return 0.0

        return co_occurrence_count / union_count

    def get_related_skills(
        self,
        skill,
        min_weight=1,
        min_similarity=0.0,
    ):

        normalized = normalize_skill(skill)

        if normalized not in self.nodes:
            return []

        related = []

        for target, details in self.edges[normalized].items():

            if details["relationship"] != "RELATED_TO":
                continue

            if details["weight"] < min_weight:
                continue

            if details["jaccard_similarity"] < min_similarity:
                continue

            related.append({
                "skill": target,
                "weight": details["weight"],
                "jaccard_similarity": round(
                    details["jaccard_similarity"], 4
                ),
            })

        return sorted(
            related,
            key=lambda item: (
                -item["jaccard_similarity"],
                -item["weight"],
                item["skill"],
            ),
        )

    def get_category_skills(self, category):

        return sorted(
            node["name"]
            for node in self.nodes.values()
            if node["type"] == "skill"
            and node["category"] == category
        )

    def get_neighbors(self, skill):

        normalized = normalize_skill(skill)

        if normalized not in self.nodes:
            return []

        return [
            {
                "target": target,
                **details,
            }
            for target, details in sorted(
                self.edges[normalized].items()
            )
        ]

    def get_graph_summary(self):

        skill_count = sum(
            1 for node in self.nodes.values()
            if node["type"] == "skill"
        )

        category_count = sum(
            1 for node in self.nodes.values()
            if node["type"] == "category"
        )

        relationship_count = sum(
            1
            for targets in self.edges.values()
            for details in targets.values()
            if details["relationship"] == "RELATED_TO"
        )

        return {
            "total_nodes": len(self.nodes),
            "skill_nodes": skill_count,
            "category_nodes": category_count,
            "directed_skill_relationships": relationship_count,
            "unique_skill_relationships": relationship_count // 2,
        }
