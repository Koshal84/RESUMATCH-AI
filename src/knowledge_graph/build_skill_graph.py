
import json
import ast

from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd

from src.knowledge_graph.skill_graph import SkillKnowledgeGraph


class DatasetSkillGraphBuilder:

    def __init__(self):
        self.graph = SkillKnowledgeGraph()

    @staticmethod
    def parse_skills(value):

        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        if not isinstance(value, str):
            return []

        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except (json.JSONDecodeError, TypeError):
            pass

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return parsed

        except (ValueError, SyntaxError):
            pass

        return [
            skill.strip()
            for skill in value.split(",")
            if skill.strip()
        ]

    def build_from_jobs(self, jobs_df, min_weight=2):

        if "clean_skills" not in jobs_df.columns:
            raise ValueError(
                "Dataset must contain clean_skills column."
            )

        co_occurrence = Counter()
        skill_frequency = Counter()

        for skills_value in jobs_df["clean_skills"]:

            skills = self.parse_skills(skills_value)

            normalized = {
                self.graph.add_skill(skill)
                for skill in skills
            }

            normalized.discard(None)

            # Count each skill once per job.
            skill_frequency.update(normalized)

            # Count each pair once per job.
            for first, second in combinations(
                sorted(normalized), 2
            ):
                co_occurrence[(first, second)] += 1

        self.graph.set_skill_frequency(skill_frequency)

        for (first, second), count in co_occurrence.items():

            if count < min_weight:
                continue

            first_frequency = skill_frequency[first]
            second_frequency = skill_frequency[second]

            similarity = self.graph.calculate_jaccard(
                first_frequency,
                second_frequency,
                count,
            )

            self.graph.add_related_skills(
                first,
                second,
                weight=count,
                similarity=similarity,
            )

        return self.graph

    def build_from_csv(self, csv_path, min_weight=2):

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        jobs_df = pd.read_csv(csv_path)

        return self.build_from_jobs(
            jobs_df,
            min_weight=min_weight,
        )


if __name__ == "__main__":

    dataset_path = "data/processed/jobs_cleaned.csv"

    builder = DatasetSkillGraphBuilder()

    graph = builder.build_from_csv(
        dataset_path,
        min_weight=2,
    )

    print("\n========== JACCARD SKILL GRAPH ==========")
    print(graph.get_graph_summary())

    print("\nTop related skills for Python:")

    for item in graph.get_related_skills(
        "Python",
        min_weight=2,
        min_similarity=0.0,
    )[:20]:
        print(item)

    print("\nStrong Python relationships (Jaccard >= 0.10):")

    for item in graph.get_related_skills(
        "Python",
        min_weight=2,
        min_similarity=0.10,
    ):
        print(item)
