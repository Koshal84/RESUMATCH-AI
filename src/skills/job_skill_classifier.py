
import re


class JobSkillClassifier:
    """
    Evidence-based job skill classifier.

    Required: explicit mandatory evidence.
    Preferred: explicit desirable evidence.
    Unclassified: ambiguous, missing, conflicting or boilerplate evidence.
    """

    REQUIRED_HEADINGS = [
        "required qualifications",
        "minimum qualifications",
        "required skills",
        "essential qualifications",
        "qualifications required",
        "knowledge and skills requirements",
    ]

    PREFERRED_HEADINGS = [
        "preferred qualifications",
        "preferred skills",
        "desired qualifications",
        "nice to have",
        "bonus qualifications",
    ]

    OTHER_HEADINGS = [
        "essential duties and responsibilities",
        "duties and responsibilities",
        "job responsibilities",
        "key responsibilities",
        "responsibilities",
        "about the role",
        "about the company",
        "about ecolab institutional",
        "what you will do",
        "what you'll do",
        "job summary",
        "position summary",
        "education and experience",
        "overview",
    ]

    REQUIRED_PATTERNS = [
        r"\bmust have\b",
        r"\bmust possess\b",
        r"\bis required\b",
        r"\bare required\b",
        r"\bwill be required\b",
        r"\brequired qualification\b",
        r"\brequired skill\b",
        r"\bminimum requirement\b",
        r"\bminimum of\b",
        r"\bat least\b",
        r"\bexperience required\b",
        r"\brequired\b",
    ]

    PREFERRED_PATTERNS = [
        r"\bpreferred\b",
        r"\bhighly desirable\b",
        r"\bdesirable\b",
        r"\bnice to have\b",
        r"\ba plus\b",
        r"\bbig plus\b",
        r"\bbonus\b",
        r"\bideally\b",
        r"\bdesired\b",
    ]

    NEGATIVE_PATTERNS = [
        r"\bnot required\b",
        r"\bnot necessarily required\b",
        r"\bnot essential\b",
        r"\bnot necessary\b",
        r"\bno requirement\b",
        r"\bnot mandatory\b",
    ]

    BOILERPLATE_PATTERNS = [
        r"does not accept.{0,150}unsolicited",
        r"unsolicited resumes",
        r"unsolicited recruiting firms",
        r"third.party recruiting firms",
        r"recruiting firms or agencies",
        r"recruiting-coordinator",
        r"staffing agencies",
    ]

    def __init__(self):
        self.heading_patterns = []

        for heading in self.REQUIRED_HEADINGS:
            self.heading_patterns.append((heading, "required"))

        for heading in self.PREFERRED_HEADINGS:
            self.heading_patterns.append((heading, "preferred"))

        for heading in self.OTHER_HEADINGS:
            self.heading_patterns.append((heading, "unclassified"))

        self.heading_patterns.sort(
            key=lambda item: len(item[0]),
            reverse=True
        )

    def _find_sections(self, description):
        text = description.lower()
        markers = []

        for heading, classification in self.heading_patterns:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(heading)
                + r"(?![a-z0-9])"
            )

            for match in re.finditer(pattern, text):
                markers.append({
                    "start": match.start(),
                    "end": match.end(),
                    "classification": classification,
                    "heading": heading
                })

        markers.sort(key=lambda item: item["start"])

        filtered = []

        for marker in markers:
            if filtered and marker["start"] < filtered[-1]["end"]:
                continue

            filtered.append(marker)

        sections = []

        for index, marker in enumerate(filtered):
            section_end = (
                filtered[index + 1]["start"]
                if index + 1 < len(filtered)
                else len(text)
            )

            sections.append({
                "start": marker["end"],
                "end": section_end,
                "classification": marker["classification"],
                "heading": marker["heading"]
            })

        return sections

    def _find_skill_positions(self, description, matched_alias):
        if not matched_alias:
            return []

        alias = str(matched_alias).strip().lower()

        if not alias:
            return []

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(alias)
            + r"(?![a-z0-9])"
        )

        return [
            match.start()
            for match in re.finditer(pattern, description.lower())
        ]

    def _get_sentence(self, text, position):
        start = position
        end = position

        while start > 0 and text[start - 1] not in ".!?;\n":
            start -= 1

        while end < len(text) and text[end] not in ".!?;\n":
            end += 1

        return text[start:end].strip()

    def _is_boilerplate(self, sentence):
        sentence = sentence.lower()

        return any(
            re.search(pattern, sentence)
            for pattern in self.BOILERPLATE_PATTERNS
        )

    def _classify_sentence(self, sentence):
        sentence = sentence.lower()

        if self._is_boilerplate(sentence):
            return "unclassified"

        if any(
            re.search(pattern, sentence)
            for pattern in self.NEGATIVE_PATTERNS
        ):
            return "unclassified"

        required = any(
            re.search(pattern, sentence)
            for pattern in self.REQUIRED_PATTERNS
        )

        preferred = any(
            re.search(pattern, sentence)
            for pattern in self.PREFERRED_PATTERNS
        )

        if required and not preferred:
            return "required"

        if preferred and not required:
            return "preferred"

        return "unclassified"

    def _classify_occurrences(self, positions, sections, text):
        classifications = []

        for position in positions:
            section_classification = "unclassified"
            heading = None

            for section in sections:
                if section["start"] <= position < section["end"]:
                    section_classification = section["classification"]
                    heading = section["heading"]
                    break

            sentence = self._get_sentence(text, position)

            # Boilerplate must never inherit a section classification.
            if self._is_boilerplate(sentence):
                classifications.append({
                    "classification": "unclassified",
                    "heading": "boilerplate exclusion",
                    "position": position,
                    "sentence": sentence
                })
                continue

            sentence_classification = self._classify_sentence(sentence)

            if sentence_classification != "unclassified":
                classification = sentence_classification
                evidence_heading = "sentence-level evidence"

            elif any(
                re.search(pattern, sentence.lower())
                for pattern in self.NEGATIVE_PATTERNS
            ):
                classification = "unclassified"
                evidence_heading = "negative sentence evidence"

            else:
                classification = section_classification
                evidence_heading = heading

            classifications.append({
                "classification": classification,
                "heading": evidence_heading,
                "position": position,
                "sentence": sentence
            })

        return classifications

    def _resolve_classification(self, occurrences):
        meaningful = {
            item["classification"]
            for item in occurrences
            if item["classification"] in {"required", "preferred"}
        }

        if len(meaningful) != 1:
            return "unclassified", None, occurrences

        classification = next(iter(meaningful))

        supporting = [
            item for item in occurrences
            if item["classification"] == classification
        ]

        return (
            classification,
            supporting[0]["heading"],
            supporting
        )

    def classify(self, description, extracted_skills):
        description = description or ""
        text = description.lower()

        sections = self._find_sections(description)

        result = {
            "required_skills": [],
            "preferred_skills": [],
            "unclassified_skills": []
        }

        evidence = []

        for item in extracted_skills or []:
            if isinstance(item, dict):
                skill_name = item.get("skill")
                category = item.get("category")
                matched_alias = item.get("matched_alias") or skill_name

            elif isinstance(item, str):
                skill_name = item
                category = None
                matched_alias = item

            else:
                continue

            if not skill_name:
                continue

            positions = self._find_skill_positions(
                description,
                matched_alias
            )

            occurrences = self._classify_occurrences(
                positions,
                sections,
                text
            )

            classification, heading, supporting = (
                self._resolve_classification(occurrences)
            )

            output_item = {
                "skill": skill_name,
                "category": category
            }

            result[classification + "_skills"].append(output_item)

            evidence_sentences = list(dict.fromkeys(
                occurrence["sentence"]
                for occurrence in supporting
                if occurrence["sentence"]
            ))

            result_evidence = {
                "skill": skill_name,
                "classification": classification,
                "evidence_heading": heading,
                "evidence_sentence": " | ".join(evidence_sentences) or None,
                "occurrence_count": len(positions)
            }

            evidence.append(result_evidence)

        result["classification_evidence"] = evidence

        result["counts"] = {
            "required": len(result["required_skills"]),
            "preferred": len(result["preferred_skills"]),
            "unclassified": len(result["unclassified_skills"])
        }

        return result
