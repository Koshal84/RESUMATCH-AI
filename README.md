# RESUMATCH AI

**An Explainable, Context-Aware Resume–Job Matching and Skill Gap Intelligence System**

RESUMATCH AI is a CPU-friendly NLP application that analyzes resumes and job descriptions, extracts relevant skills, ranks job opportunities, identifies skill gaps, and provides evidence-based explanations for matching results.

The system combines traditional text-based matching with semantic similarity to explore how different approaches perform on resume–job retrieval tasks.

## Key Features

- **Resume Parsing:** Extract text from PDF and DOCX resumes.
- **Skill Extraction:** Identify technical and professional skills using a structured skill taxonomy.
- **Text Preprocessing:** Clean and normalize resume and job-description text.
- **Job Matching:** Rank job descriptions using TF-IDF, semantic similarity, and hybrid matching.
- **Explainable Matching:** Provide evidence-based explanations for matching results.
- **Skill Gap Analysis:** Identify matched, missing, and additional skills.
- **Skill Knowledge Graph:** Explore relationships between skills based on their co-occurrence in job descriptions.
- **Model Evaluation:** Compare matching approaches using Precision@5, Recall@5, and Mean Reciprocal Rank (MRR).
- **Interactive Dashboard:** Explore the system through a Streamlit web application.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and analysis |
| Scikit-learn | TF-IDF and machine learning |
| Sentence Transformers | Semantic embeddings |
| Streamlit | Interactive web application |
| PyPDF | PDF text extraction |
| Python-docx | DOCX text extraction |
| PyYAML | Configuration management |
| Pytest | Automated testing |

## System Architecture

The application follows a modular architecture:

```text
                 RESUMATCH AI
                       |
          +------------+------------+
          |                         |
     Resume Input               Job Dataset
          |                         |
          +------------+------------+
                       |
               Text Preprocessing
                       |
                 Skill Extraction
                       |
          +------------+------------+
          |            |            |
        TF-IDF      Semantic     Skill Analysis
        Model        Model           |
          |            |        Skill Taxonomy
          +------+-----+             |
                 |                   |
           Hybrid Matching     Skill Gap Analysis
                 |
          Job Ranking Results
                 |
          Explainability Engine
                 |
          Streamlit Dashboard
                 |
          Model Evaluation
                 |
           Skill Knowledge Graph
```

## Repository Structure

```text
RESUMATCH-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- configs/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- labels/
|   |-- external/
|
|-- docs/
|
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- explainability/
|   |-- knowledge_graph/
|   |-- matching/
|   |-- models/
|   |-- preprocessing/
|   |-- skills/
|
|-- tests/
|
|-- .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RESUMATCH-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Application Modules

### 1. Resume Analyzer

- Upload PDF or DOCX resumes.
- Extract and process resume text.
- Identify skills using the skill extraction module.
- Display extracted skills and supporting text evidence.

### 2. Job Matching

- Compare resume content with available job descriptions.
- Generate TF-IDF and semantic similarity scores.
- Combine model outputs through hybrid matching.
- Rank relevant job opportunities.

### 3. Skill Gap Analysis

- Identify skills shared by a candidate and a job.
- Display job-listed skills missing from the resume.
- Identify additional candidate skills.
- Present skill matching information.

Note: The current dataset does not provide verified required/preferred labels for every job skill. Therefore, the application treats these as job-listed skills rather than confirmed mandatory requirements.

### 4. Explainability Engine

- Generate explanations for matching results.
- Show relevant evidence from resume and job-description text.
- Help users understand why a particular match was produced.

### 5. Skill Knowledge Graph

- Represent skills and categories as graph nodes.
- Model skill relationships using co-occurrence information.
- Explore related skills and skill categories.

A graph relationship indicates that skills co-occurred in job descriptions. It does not necessarily mean that the skills are equivalent.

### 6. Model Evaluation

Compare the performance of TF-IDF, semantic, and hybrid matching models using ranking metrics.

## Model Evaluation Results

The following results were obtained using a holdout evaluation of 100 candidates.

| Model | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| TF-IDF | 0.142 | 0.091787 | 0.254333 |
| Semantic | 0.064 | 0.035539 | 0.168667 |
| Hybrid 70:30 | 0.120 | 0.080451 | 0.255000 |

### Key Observations

- TF-IDF achieved the highest Precision@5 and Recall@5 in this evaluation.
- Hybrid matching achieved a slightly higher MRR than TF-IDF.
- Semantic matching alone produced lower scores across the three reported metrics.

These results indicate that traditional lexical matching performed competitively on this dataset. Hybrid matching showed a small advantage in reciprocal rank, but did not outperform TF-IDF on every metric.

### Evaluation Limitations

- Evaluation uses recorded positive relevance labels.
- Unlisted resume–job pairs are unknown, not confirmed negative matches.
- Results are specific to the dataset and evaluation split.
- These metrics do not measure real-world hiring success or employment outcomes.

## Automated Testing

The project includes automated tests for important components.

Run the test suite using:

```bash
python -m pytest -v
```

**Current test result:**

```text
29 passed
```

The tests cover:

- Dataset validation
- Skill extraction
- Evidence generation
- Explainability
- Skill knowledge graph operations

## System Requirements

- Python 3.13.4 (tested environment)
- 16 GB RAM recommended
- CPU-supported execution
- NVIDIA GPU is not required

Semantic embedding models may need to be downloaded the first time they are used.

## Data Privacy and Responsible Use

Resume documents can contain personally identifiable information.

- Avoid publishing real resumes.
- Do not commit personal information, API keys, or confidential files.
- Verify dataset licenses and redistribution permissions before publishing datasets.
- Prefer synthetic or consented examples for public demonstrations.
- Similarity scores should not be interpreted as hiring probabilities or measures of candidate quality.

RESUMATCH AI is an experimental decision-support and learning application. It is not intended to make automated employment decisions.

## Current Project Status

- [x] Dataset preprocessing and validation
- [x] Skill extraction and taxonomy
- [x] TF-IDF matching model
- [x] Semantic matching model
- [x] Hybrid matching model
- [x] Skill gap analysis
- [x] Explainability engine
- [x] Skill knowledge graph
- [x] Streamlit application
- [x] Model evaluation integration
- [x] Automated testing
- [ ] Final documentation and release verification

## Future Improvements

- Improve skill extraction and normalization.
- Expand evaluation using manually reviewed relevance labels.
- Improve ranking through additional experiments.
- Add more robust resume and job-description formats.
- Improve documentation and deployment support.

## License

See the repository's `LICENSE` file for the applicable source-code license.

Third-party datasets and pretrained models remain subject to their respective licenses and terms.
