
import re

from src.skills.skill_taxonomy import (
    SKILL_TAXONOMY,
    get_skill_category
)


class SkillExtractor:

    def __init__(self):

        self.alias_to_skill = {}

        excluded_aliases = {"py", "js", "ts", "ml", "dl", "ai"}

        # 1. Build skill alias dictionary
        for canonical_name, details in SKILL_TAXONOMY.items():

            if canonical_name in {
                "Sourcing",
                "Customer Relationship Management"
            }:
                aliases = details["aliases"]
            else:
                aliases = [canonical_name] + details["aliases"]

            for alias in aliases:

                normalized_alias = alias.strip().lower()

                if (
                    len(normalized_alias) < 3
                    and normalized_alias not in {"c++", "c#"}
                ):
                    continue

                if normalized_alias in excluded_aliases:
                    continue

                self.alias_to_skill[normalized_alias] = canonical_name

        # Match longer aliases first
        sorted_aliases = sorted(
            self.alias_to_skill.keys(),
            key=len,
            reverse=True
        )

        alternatives = "|".join(
            re.escape(alias) for alias in sorted_aliases
        )

        self.pattern = re.compile(
            r"(?<!\w)(?P<skill>" + alternatives + r")(?!\w)",
            re.IGNORECASE
        )

        # 2. Sourcing detection
        self.recruitment_context = re.compile(
            r"\b(candidate|candidates|applicant|applicants|"
            r"recruitment|recruiting|recruit|hiring|"
            r"talent acquisition|job candidates)\b",
            re.IGNORECASE
        )

        self.sourcing_pattern = re.compile(
            r"\bsourcing\b",
            re.IGNORECASE
        )

        # 3. CRM relationship detection
        self.relationship_pattern = re.compile(
            r"\brelationships?\b",
            re.IGNORECASE
        )

        self.customer_client_pattern = re.compile(
            r"\b(customers?|clients?)\b",
            re.IGNORECASE
        )

        self.relationship_action_pattern = re.compile(
            r"\b(build|building|built|develop|developing|developed|"
            r"maintain|maintaining|maintained|manage|managing|managed|"
            r"establish|establishing|established|strengthen|"
            r"strengthening|strengthened|cultivate|cultivating|"
            r"cultivated|nurture|nurturing|nurtured|"
            r"grow|growing|grown|retain|retaining|retained)\b",
            re.IGNORECASE
        )

        # 4. Direct CRM software detection
        self.crm_software_pattern = re.compile(
            r"\b(salesforce\s+crm|salesforce|hubspot|"
            r"zoho\s+crm|microsoft\s+dynamics(?:\s+365)?|"
            r"sugarcrm|pipedrive|freshsales)\b",
            re.IGNORECASE
        )

        # 5. Resume section headings
        self.section_pattern = re.compile(
            r"\b(?:"
            r"technical skills|key skills|core competencies|skills|"
            r"academic background|education|"
            r"professional experience|work experience|experience|"
            r"employment history|"
            r"academic projects|personal projects|projects|"
            r"certifications|certificates|"
            r"professional summary|summary|career objective|objective|"
            r"target position|"
            r"achievements|responsibilities|"
            r"tools and technologies"
            r")\s*:",
            re.IGNORECASE
        )

        # Sections that commonly contain comma-separated skill lists
        self.skill_list_sections = {
            "skills",
            "technical skills",
            "key skills",
            "core competencies",
            "tools and technologies"
        }

    def _get_evidence_sentence(self, text, position):

        headings = list(self.section_pattern.finditer(text))

        previous_heading = None
        next_heading = None

        # Identify the section containing the detected skill
        for heading in headings:

            if heading.start() <= position:
                previous_heading = heading

            elif heading.start() > position:
                next_heading = heading
                break

        section_start = (
            previous_heading.end()
            if previous_heading
            else 0
        )

        section_end = (
            next_heading.start()
            if next_heading
            else len(text)
        )

        section_name = (
            previous_heading.group(0).rstrip(":").strip().lower()
            if previous_heading
            else ""
        )

        # 1. Handle comma-separated skill-list sections
        if section_name in self.skill_list_sections:

            start = position
            end = position

            # Move backwards to the previous list separator
            while (
                start > section_start
                and text[start - 1] not in ",;\n"
            ):
                start -= 1

            # Move forwards to the next list separator
            while (
                end < section_end
                and text[end] not in ",;\n"
            ):
                end += 1

            evidence = text[start:end].strip()

            if previous_heading:
                return (
                    f"{previous_heading.group(0).strip()} "
                    f"{evidence}"
                ).strip()

            return evidence

        # 2. Handle sentence-based evidence in other sections
        start = section_start
        end = section_end

        for i in range(position - 1, section_start - 1, -1):

            if text[i] in ".!?;\n":
                start = i + 1
                break

        for i in range(position, section_end):

            if text[i] in ".!?;\n":
                end = i + 1
                break

        evidence = text[start:end].strip()

        # Fallback
        if not evidence:
            return text[position:position + 100].strip()

        return evidence

    def _add_skill(
        self,
        found_skills,
        canonical_name,
        matched_alias,
        text,
        start_position,
        end_position
    ):

        if canonical_name not in found_skills:

            found_skills[canonical_name] = {
                "skill": canonical_name,
                "category": get_skill_category(canonical_name),
                "matched_alias": matched_alias,
                "start_position": start_position,
                "end_position": end_position,
                "evidence_sentence": self._get_evidence_sentence(
                    text,
                    start_position
                )
            }

    def extract_skills(self, text):

        if not isinstance(text, str) or not text.strip():
            return []

        found_skills = {}

        # 1. Taxonomy-based extraction
        for match in self.pattern.finditer(text):

            matched_alias = match.group("skill").lower()

            canonical_name = self.alias_to_skill.get(matched_alias)

            if not canonical_name:
                continue

            self._add_skill(
                found_skills,
                canonical_name,
                matched_alias,
                text,
                match.start(),
                match.end()
            )

            # Microsoft SQL Server also implies SQL
            if canonical_name == "Microsoft SQL Server":

                sql_match = re.search(
                    r"(?<!\w)sql(?!\w)",
                    match.group(0),
                    re.IGNORECASE
                )

                if sql_match:
                    sql_start = match.start() + sql_match.start()
                    sql_end = match.start() + sql_match.end()

                else:
                    sql_start = match.start()
                    sql_end = match.end()

                self._add_skill(
                    found_skills,
                    "SQL",
                    "sql",
                    text,
                    sql_start,
                    sql_end
                )

        # 2. Context-aware Sourcing detection
        for match in self.sourcing_pattern.finditer(text):

            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)

            context = text[start:end]

            if self.recruitment_context.search(context):

                self._add_skill(
                    found_skills,
                    "Sourcing",
                    "sourcing",
                    text,
                    match.start(),
                    match.end()
                )

                break

        # 3. Direct CRM software detection
        software_match = self.crm_software_pattern.search(text)

        if software_match:

            self._add_skill(
                found_skills,
                "Customer Relationship Management",
                software_match.group(0).lower(),
                text,
                software_match.start(),
                software_match.end()
            )

        # 4. Context-aware CRM relationship detection
        for match in self.relationship_pattern.finditer(text):

            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)

            context = text[start:end]

            has_customer_or_client = bool(
                self.customer_client_pattern.search(context)
            )

            has_relationship_action = bool(
                self.relationship_action_pattern.search(context)
            )

            if has_customer_or_client and has_relationship_action:

                self._add_skill(
                    found_skills,
                    "Customer Relationship Management",
                    "customer/client relationship context",
                    text,
                    match.start(),
                    match.end()
                )

                break

        return sorted(
            found_skills.values(),
            key=lambda item: item["skill"].lower()
        )
