from pathlib import Path
from io import BytesIO
import json

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document

from src.skills.skill_extractor import SkillExtractor
from src.skills.skill_gap_analyzer import SkillGapAnalyzer
from src.matching.matching_service import JobMatchingService
from src.knowledge_graph.build_skill_graph import DatasetSkillGraphBuilder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

JOBS_FILE = DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_FILE = DATA_DIR / "candidates_cleaned.csv"

st.set_page_config(
    page_title="RESUMATCH AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


jobs_df = load_csv(JOBS_FILE)
candidates_df = load_csv(CANDIDATES_FILE)


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def load_skill_extractor():
    return SkillExtractor()


@st.cache_resource
def load_skill_gap_analyzer():
    return SkillGapAnalyzer()


@st.cache_resource(show_spinner=False)
def load_matching_service():
    return JobMatchingService(jobs_df)


@st.cache_resource(show_spinner=False)
def load_knowledge_graph(jobs_data):
    builder = DatasetSkillGraphBuilder()
    graph_df = pd.DataFrame(jobs_data, columns=["clean_skills"])

    return builder.build_from_jobs(
        graph_df,
        min_weight=2,
    )


skill_extractor = load_skill_extractor()
skill_gap_analyzer = load_skill_gap_analyzer()


# --------------------------------------------------
# RESUME TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)


def extract_docx_text(file_bytes):
    document = Document(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    table_text = []

    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                table_text.append(" | ".join(cells))

    return "\n".join(paragraphs + table_text)


def extract_resume_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_bytes)

    if extension == ".docx":
        return extract_docx_text(file_bytes)

    raise ValueError("Unsupported file format.")


# --------------------------------------------------
# JOB SKILL PARSING
# --------------------------------------------------

def parse_job_skills(raw_skills):
    if isinstance(raw_skills, list):
        return raw_skills

    if isinstance(raw_skills, str):
        parsed = json.loads(raw_skills)

        if isinstance(parsed, list):
            return parsed

    raise ValueError("Job skills must be stored as a JSON list.")


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #312e81);
        border: 1px solid #3730a3;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #c7d2fe;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🎯 RESUMATCH AI")
    st.caption("Intelligent Resume & Job Matching")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Resume Analyzer",
            "Job Matching",
            "Skill Gap Analysis",
            "Knowledge Graph",
            "Model Evaluation",
        ],
    )

    st.divider()
    st.caption("RESUMATCH AI · Phase 12")
    st.caption("Explainable AI-powered recruitment intelligence")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to RESUMATCH AI</h1>
            <p>
                Explainable, context-aware resume matching
                and skill gap intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Available Jobs", f"{len(jobs_df):,}")

    with col2:
        st.metric("Candidate Profiles", f"{len(candidates_df):,}")

    with col3:
        st.metric("Matching Models", "3")

    st.divider()
    st.subheader("System Components")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Intelligent Matching")
        st.write(
            "TF-IDF, semantic similarity and hybrid ranking "
            "for candidate-job retrieval."
        )
        st.markdown("### Skill Intelligence")
        st.write(
            "Skill extraction, taxonomy normalization, "
            "job skill classification and gap analysis."
        )

    with c2:
        st.markdown("### Explainable Recommendations")
        st.write(
            "Evidence-based matching scores and transparent "
            "skill comparisons."
        )
        st.markdown("### Skill Knowledge Graph")
        st.write(
            "Explore relationships between skills using "
            "job-description co-occurrence."
        )

    st.divider()
    st.subheader("Dataset Preview")

    if not jobs_df.empty:
        st.dataframe(jobs_df.head(5), use_container_width=True)
    else:
        st.warning(
            "Jobs dataset not found: data/processed/jobs_cleaned.csv"
        )

    if not candidates_df.empty:
        st.caption("Candidate dataset preview")
        st.dataframe(candidates_df.head(5), use_container_width=True)
    else:
        st.warning(
            "Candidate dataset not found: "
            "data/processed/candidates_cleaned.csv"
        )


# --------------------------------------------------
# RESUME ANALYZER
# --------------------------------------------------

elif page == "Resume Analyzer":

    st.title("📄 Resume Analyzer")
    st.write(
        "Upload a PDF/DOCX resume or paste resume text "
        "to identify skills and supporting evidence."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        key="analyzer_upload",
        help="Supported formats: PDF and DOCX.",
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: {uploaded_file.name}")

    st.divider()

    resume_text_input = st.text_area(
        "Or paste resume text",
        height=220,
        placeholder="Paste your resume text here...",
        key="analyzer_text",
    )

    analyze_button = st.button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:
        try:
            if uploaded_file is not None:
                with st.spinner("Reading uploaded resume..."):
                    resume_text = extract_resume_text(uploaded_file)
            elif resume_text_input.strip():
                resume_text = resume_text_input
            else:
                st.warning(
                    "Please upload a resume or paste resume text."
                )
                resume_text = ""

            if resume_text:
                if not resume_text.strip():
                    st.warning(
                        "No readable text was found. "
                        "This PDF may be scanned or image-based."
                    )
                else:
                    with st.spinner("Extracting skills..."):
                        extracted_skills = (
                            skill_extractor.extract_skills(resume_text)
                        )

                    st.success("Resume analysis completed!")

                    with st.expander("View extracted resume text"):
                        st.text(resume_text)

                    st.divider()
                    st.subheader("Analysis Overview")

                    total_skills = len(extracted_skills)
                    categories = {}

                    for item in extracted_skills:
                        category = item["category"]
                        categories[category] = (
                            categories.get(category, 0) + 1
                        )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Skills Identified", total_skills)

                    with col2:
                        st.metric("Skill Categories", len(categories))

                    st.divider()

                    if extracted_skills:
                        st.subheader("Extracted Skills")
                        skills_df = pd.DataFrame(extracted_skills)

                        display_columns = [
                            "skill",
                            "category",
                            "matched_alias",
                            "evidence_sentence",
                        ]

                        st.dataframe(
                            skills_df[display_columns],
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.divider()
                        st.subheader("Category-wise Skill Distribution")

                        category_df = pd.DataFrame(
                            list(categories.items()),
                            columns=["Category", "Number of Skills"],
                        )

                        st.bar_chart(
                            category_df.set_index("Category")
                        )

                        st.divider()
                        st.subheader("Detected Skill List")

                        for category, count in sorted(categories.items()):
                            st.markdown(f"**{category}** — {count}")

                            category_skills = [
                                item["skill"]
                                for item in extracted_skills
                                if item["category"] == category
                            ]

                            st.write(", ".join(category_skills))
                    else:
                        st.info(
                            "No skills were identified. "
                            "Try providing more resume content."
                        )

        except Exception as error:
            st.error(f"Could not process the resume: {error}")


# --------------------------------------------------
# JOB MATCHING
# --------------------------------------------------

elif page == "Job Matching":

    st.title("🎯 Intelligent Job Matching")
    st.write(
        "Find relevant job opportunities using TF-IDF "
        "and semantic similarity with hybrid ranking."
    )
    st.divider()

    if jobs_df.empty:
        st.error("Jobs dataset is missing or empty.")
        st.stop()

    required_columns = {
        "job_id",
        "job_text",
        "title",
        "description",
    }

    missing_columns = required_columns - set(jobs_df.columns)

    if missing_columns:
        st.error(
            f"Jobs dataset is missing columns: {sorted(missing_columns)}"
        )
        st.stop()

    st.subheader("Upload Candidate Resume")

    matching_file = st.file_uploader(
        "Upload PDF or DOCX resume",
        type=["pdf", "docx"],
        key="matching_upload",
    )

    matching_text = st.text_area(
        "Or paste your resume text",
        height=220,
        placeholder="Paste your resume content here...",
        key="matching_resume_text",
    )

    top_k = st.selectbox(
        "Number of job recommendations",
        [5, 10, 15, 20],
        index=1,
    )

    match_button = st.button(
        "Find Matching Jobs",
        type="primary",
        use_container_width=True,
    )

    if match_button:
        try:
            if matching_file is not None:
                with st.spinner("Reading resume..."):
                    resume_text = extract_resume_text(matching_file)
            else:
                resume_text = matching_text

            if not resume_text or not resume_text.strip():
                st.warning(
                    "Please upload a resume or paste resume text."
                )
            else:
                with st.spinner(
                    "Preparing matching models and ranking jobs. "
                    "The first run may take some time..."
                ):
                    matching_service = load_matching_service()
                    results_df = matching_service.match_resume(
                        resume_text=resume_text,
                        top_k=top_k,
                    )

                if results_df.empty:
                    st.warning("No matching jobs were found.")
                else:
                    st.success("Job matching completed!")
                    st.divider()
                    st.subheader("Matching Overview")

                    best_score = results_df.iloc[0]["hybrid_score"]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Jobs Analyzed", f"{len(jobs_df):,}")

                    with col2:
                        st.metric("Recommendations", len(results_df))

                    with col3:
                        st.metric(
                            "Top Hybrid Similarity",
                            f"{best_score * 100:.2f}%",
                        )

                    st.caption(
                        "Similarity scores indicate text relevance, "
                        "not the probability of getting hired."
                    )

                    st.divider()
                    st.subheader("Top Job Recommendations")

                    chart_df = results_df[
                        ["title", "hybrid_score"]
                    ].copy()
                    chart_df["hybrid_score"] *= 100
                    st.bar_chart(chart_df.set_index("title"))

                    st.divider()

                    for index, row in results_df.iterrows():
                        score = row["hybrid_score"] * 100

                        with st.expander(
                            f"#{index + 1} — {row['title']} | "
                            f"Similarity: {score:.2f}%",
                            expanded=(index == 0),
                        ):
                            st.markdown("**Job Description**")
                            st.write(
                                row.get(
                                    "description",
                                    "Description unavailable.",
                                )
                            )

                            score_col1, score_col2, score_col3 = (
                                st.columns(3)
                            )

                            with score_col1:
                                st.metric(
                                    "Hybrid Similarity",
                                    f"{score:.2f}%",
                                )

                            with score_col2:
                                st.metric(
                                    "TF-IDF Similarity",
                                    f"{row['tfidf_score'] * 100:.2f}%",
                                )

                            with score_col3:
                                st.metric(
                                    "Semantic Similarity",
                                    f"{row['semantic_score'] * 100:.2f}%",
                                )

                            st.caption(f"Job ID: {row['job_id']}")

                    st.divider()
                    st.subheader("Detailed Matching Results")

                    display_df = results_df[
                        [
                            "job_id",
                            "title",
                            "tfidf_score",
                            "semantic_score",
                            "hybrid_score",
                        ]
                    ].copy()

                    display_df["tfidf_score"] *= 100
                    display_df["semantic_score"] *= 100
                    display_df["hybrid_score"] *= 100

                    display_df = display_df.rename(
                        columns={
                            "tfidf_score": "TF-IDF (%)",
                            "semantic_score": "Semantic (%)",
                            "hybrid_score": "Hybrid (%)",
                        }
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.error(f"Job matching failed: {error}")


# --------------------------------------------------
# SKILL GAP ANALYSIS
# --------------------------------------------------

elif page == "Skill Gap Analysis":

    st.title("📊 Skill Gap Analysis")
    st.write(
        "Compare candidate skills against the skills listed "
        "in a selected job description."
    )

    st.info(
        "The current dataset does not distinguish required "
        "and preferred skills. All extracted job skills are "
        "treated as job-listed skills."
    )

    if jobs_df.empty:
        st.error("Jobs dataset is missing or empty.")
        st.stop()

    required_columns = {
        "job_id",
        "title",
        "description",
        "clean_skills",
    }

    missing_columns = required_columns - set(jobs_df.columns)

    if missing_columns:
        st.error(
            f"Jobs dataset is missing columns: {sorted(missing_columns)}"
        )
        st.stop()

    st.subheader("Select Target Job")

    selected_job_index = st.selectbox(
        "Choose a job role",
        jobs_df.index.tolist(),
        format_func=lambda idx: (
            f"{jobs_df.loc[idx, 'title']} "
            f"(ID: {jobs_df.loc[idx, 'job_id']})"
        ),
        key="gap_job_selection",
    )

    selected_job = jobs_df.loc[selected_job_index]
    st.markdown(f"**Selected Role:** {selected_job['title']}")

    with st.expander("View Job Description"):
        st.write(selected_job["description"])

    st.divider()
    st.subheader("Upload Candidate Resume")

    gap_file = st.file_uploader(
        "Upload PDF or DOCX resume",
        type=["pdf", "docx"],
        key="gap_resume_upload",
    )

    gap_resume_text = st.text_area(
        "Or paste resume text",
        height=200,
        placeholder="Paste candidate resume text here...",
        key="gap_resume_text",
    )

    analyze_gap_button = st.button(
        "Analyze Skill Gap",
        type="primary",
        use_container_width=True,
    )

    if analyze_gap_button:
        try:
            if gap_file is not None:
                with st.spinner("Reading resume..."):
                    resume_text = extract_resume_text(gap_file)
            else:
                resume_text = gap_resume_text

            if not resume_text or not resume_text.strip():
                st.warning(
                    "Please upload a resume or paste resume text."
                )
            else:
                with st.spinner("Extracting candidate skills..."):
                    candidate_skills = (
                        skill_extractor.extract_skills(resume_text)
                    )

                job_skills = parse_job_skills(
                    selected_job["clean_skills"]
                )

                if not job_skills:
                    st.warning(
                        "No job skills are available for this role."
                    )
                else:
                    with st.spinner("Analyzing skill gaps..."):
                        results = skill_gap_analyzer.analyze(
                            candidate_skills=candidate_skills,
                            job_skills=job_skills,
                        )

                    st.success("Skill gap analysis completed!")
                    st.divider()
                    st.subheader("Skill Match Overview")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Overall Skill Match",
                            f"{results['overall_match_percentage']}%",
                        )

                    with col2:
                        st.metric(
                            "Matched Skills",
                            results["matched_required_count"],
                        )

                    with col3:
                        st.metric(
                            "Missing Job-listed Skills",
                            results["missing_required_count"],
                        )

                    st.divider()
                    st.subheader("Skill Match Progress")

                    progress_value = max(
                        0.0,
                        min(
                            1.0,
                            results["overall_match_percentage"] / 100,
                        ),
                    )
                    st.progress(progress_value)

                    st.caption(
                        "This percentage is the proportion of listed "
                        "job skills found in the resume. It is not a "
                        "hiring probability."
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("✅ Matched Skills")
                        matched = results["matched_required_skills"]

                        if matched:
                            for skill in matched:
                                st.write(f"✓ {skill}")
                        else:
                            st.info("No matching skills identified.")

                    with col2:
                        st.subheader("❌ Missing Job-listed Skills")
                        missing = results["missing_required_skills"]

                        if missing:
                            for skill in missing:
                                st.write(f"• {skill}")
                        else:
                            st.success("No missing listed skills!")

                    st.divider()
                    st.subheader("Additional Candidate Skills")

                    additional = results.get(
                        "extra_candidate_skills",
                        [],
                    )

                    if additional:
                        st.write(", ".join(additional))
                    else:
                        st.info(
                            "No additional candidate skills identified."
                        )

        except Exception as error:
            st.error(f"Skill gap analysis failed: {error}")


# --------------------------------------------------
# KNOWLEDGE GRAPH
# --------------------------------------------------

elif page == "Knowledge Graph":

    st.title("🧠 Skill Knowledge Graph")
    st.write(
        "Explore relationships between technical skills "
        "based on their co-occurrence in job descriptions."
    )

    st.info(
        "A relationship indicates that two skills appeared "
        "together in job descriptions. It does not mean the "
        "skills are equivalent or interchangeable."
    )

    if jobs_df.empty:
        st.error("Jobs dataset is missing or empty.")
        st.stop()

    if "clean_skills" not in jobs_df.columns:
        st.error("The jobs dataset must contain clean_skills.")
        st.stop()

    try:
        graph_input = tuple(
            jobs_df["clean_skills"].fillna("").astype(str).tolist()
        )

        with st.spinner("Building skill knowledge graph..."):
            graph = load_knowledge_graph(graph_input)

        summary = graph.get_graph_summary()

        st.subheader("Graph Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Nodes", summary["total_nodes"])

        with col2:
            st.metric("Skill Nodes", summary["skill_nodes"])

        with col3:
            st.metric("Categories", summary["category_nodes"])

        with col4:
            st.metric(
                "Unique Relationships",
                summary["unique_skill_relationships"],
            )

        st.caption(
            f"Directed skill relationships: "
            f"{summary['directed_skill_relationships']:,}"
        )

        st.divider()
        st.subheader("Explore a Skill")

        available_skills = sorted(
            node["name"]
            for node in graph.nodes.values()
            if node["type"] == "skill"
        )

        if not available_skills:
            st.warning("No skills were found in the graph.")
        else:
            selected_skill = st.selectbox(
                "Search and select a skill",
                available_skills,
                index=(
                    available_skills.index("python")
                    if "python" in available_skills
                    else 0
                ),
                key="graph_skill_selection",
            )

            min_weight = st.slider(
                "Minimum co-occurrence count",
                min_value=1,
                max_value=20,
                value=2,
                help=(
                    "Only show relationships that appeared together "
                    "at least this many times."
                ),
            )

            min_similarity = st.slider(
                "Minimum Jaccard similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            )

            related = graph.get_related_skills(
                selected_skill,
                min_weight=min_weight,
                min_similarity=min_similarity,
            )

            st.divider()
            st.subheader(f"Skills Related to {selected_skill}")
            st.metric("Related Skills Found", len(related))

            if related:
                related_df = pd.DataFrame(related)

                related_df = related_df.rename(
                    columns={
                        "skill": "Related Skill",
                        "weight": "Co-occurrence Count",
                        "jaccard_similarity": "Jaccard Similarity",
                    }
                )

                st.dataframe(
                    related_df,
                    use_container_width=True,
                    hide_index=True,
                )

                chart_df = related_df.head(15).copy()
                st.subheader("Top Related Skills")

                st.bar_chart(
                    chart_df.set_index("Related Skill")[
                        "Co-occurrence Count"
                    ]
                )
            else:
                st.info(
                    "No related skills found with these filters. "
                    "Try reducing the minimum weight or similarity."
                )

            st.divider()
            st.subheader("Skill Category")

            skill_details = graph.nodes.get(selected_skill, {})
            category = skill_details.get("category", "Unknown")
            st.metric("Category", category)

            category_skills = graph.get_category_skills(category)
            st.write(f"Skills in this category: {len(category_skills)}")
            st.write(", ".join(category_skills))

            st.divider()
            st.subheader("Relationship Details")

            neighbors = graph.get_neighbors(selected_skill)

            if neighbors:
                neighbors_df = pd.DataFrame(neighbors)
                st.dataframe(
                    neighbors_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No graph relationships are available.")

    except Exception as error:
        st.error(f"Knowledge Graph could not be loaded: {error}")


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")
    st.write(
        "Compare TF-IDF, Semantic and Hybrid models "
        "using the saved holdout evaluation results."
    )

    results_file = DATA_DIR / "holdout_evaluation_results.csv"

    if not results_file.exists():
        st.warning(
            "Evaluation results not found. Run "
            "python -m src.evaluation.evaluate_holdout "
            "from the project root, then refresh this page."
        )
        st.stop()

    try:
        results_df = pd.read_csv(results_file)

        required_columns = {
            "model",
            "evaluated_candidates",
            "precision@5",
            "recall@5",
            "mrr",
        }

        missing_columns = required_columns - set(results_df.columns)

        if missing_columns:
            st.error(
                f"Evaluation file is missing columns: "
                f"{sorted(missing_columns)}"
            )
            st.stop()

        if results_df.empty:
            st.warning("The evaluation results file is empty.")
            st.stop()

        metric_columns = ["precision@5", "recall@5", "mrr"]
        for column in metric_columns:
            results_df[column] = pd.to_numeric(
                results_df[column], errors="coerce"
            )

        if results_df[metric_columns].isna().any().any():
            st.error("Some evaluation metric values are missing or invalid.")
            st.stop()

        st.success("Holdout evaluation results loaded successfully.")

        st.divider()
        st.subheader("Evaluation Overview")

        total_evaluated = int(results_df["evaluated_candidates"].max())
        best_precision = results_df.loc[
            results_df["precision@5"].idxmax()
        ]
        best_recall = results_df.loc[
            results_df["recall@5"].idxmax()
        ]
        best_mrr = results_df.loc[
            results_df["mrr"].idxmax()
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Evaluated Candidates", total_evaluated)

        with col2:
            st.metric(
                "Best Precision@5",
                str(best_precision["model"]),
                f"{best_precision['precision@5']:.4f}",
            )

        with col3:
            st.metric(
                "Best Recall@5",
                str(best_recall["model"]),
                f"{best_recall['recall@5']:.4f}",
            )

        with col4:
            st.metric(
                "Best MRR",
                str(best_mrr["model"]),
                f"{best_mrr['mrr']:.4f}",
            )

        st.divider()
        st.subheader("Model Performance Comparison")

        metric_labels = {
            "precision@5": "Precision@5",
            "recall@5": "Recall@5",
            "mrr": "Mean Reciprocal Rank (MRR)",
        }

        selected_metric = st.selectbox(
            "Choose a metric to compare",
            options=list(metric_labels.keys()),
            format_func=lambda value: metric_labels[value],
        )

        comparison_df = results_df.set_index("model")[
            [selected_metric]
        ].rename(columns={selected_metric: metric_labels[selected_metric]})

        st.bar_chart(comparison_df, use_container_width=True)

        st.divider()
        st.subheader("Detailed Evaluation Results")

        display_df = results_df.copy()
        for column in metric_columns:
            display_df[column] = display_df[column].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("What do these metrics mean?")

        with st.expander("Precision@5"):
            st.write(
                "The proportion of the top five recommended jobs "
                "that are recorded as relevant matches."
            )

        with st.expander("Recall@5"):
            st.write(
                "The proportion of all recorded relevant jobs for a "
                "candidate that appear in the top five recommendations."
            )

        with st.expander("Mean Reciprocal Rank (MRR)"):
            st.write(
                "Measures how highly the first recorded relevant job "
                "is ranked. A higher value means the first relevant "
                "result tends to appear nearer the top."
            )

        st.divider()
        st.subheader("Model Performance Insights")

        st.write(f"**Best Precision@5:** {best_precision['model']}")
        st.write(f"**Best Recall@5:** {best_recall['model']}")
        st.write(f"**Best MRR:** {best_mrr['model']}")

        st.info(
            "Interpretation: TF-IDF leads on Precision@5 and Recall@5 "
            "in this holdout run, while Hybrid 70:30 has the highest "
            "MRR by a very small margin. No model is best on every metric."
        )

        st.caption(
            "Evaluation is based on recorded positive match labels. "
            "Unlisted candidate-job pairs are unknown, not confirmed "
            "negative examples. These metrics measure retrieval against "
            "the available labels; they are not hiring probabilities."
        )

    except Exception as error:
        st.error(f"Could not load evaluation results: {error}")
