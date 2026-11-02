# Phase 1: Data selection, objectives and methodology

## Project objective

Build a CPU-friendly, explainable retrieval system that ranks job descriptions for a resume (and later supports the reverse direction). It is a portfolio/learning tool, not a hiring decision system.

## Research questions translated into engineering objectives

1. Can contextual sentence embeddings retrieve relevant job descriptions when exact keywords differ?
2. Does combining lexical, semantic and skill-coverage signals improve ranking over a TF-IDF baseline?
3. Can every suggested match be accompanied by traceable evidence from the resume and job description?
4. How stable is ranking when documents are shortened, reordered, or contain terminology variants?

## Dataset selection

### Primary prototype candidate: Hugging Face `nonameee12233/job-resume-matching

The dataset card describes:
- `cv.csv`: 9,544 candidate records
- `job.csv`: 1,167 job records
- `matches.csv`: 122,560 candidate-job compatibility records with scoring components

Why consider it:
- It is structured around the resume–job matching task.
- It provides candidate, job, and pair-level records, which can support retrieval experiments if its labels and provenance withstand inspection.

Important caveats:
- Dataset-card descriptions are not a substitute for auditing the actual files.
- Before relying on its compatibility labels, inspect how they were generated, whether there is overlap/leakage, what each score means, and what license/redistribution terms apply.
- Do not treat compatibility scores as objective ground truth without validating their meaning.
- Keep the original downloaded data out of Git unless its license explicitly permits redistribution.

### Skills ontology: European Commission ESCO

Use ESCO as a vocabulary/ontology for occupations and skills, not as resume/job relevance ground truth. Download a consistent English CSV package and record the exact version and access date. The official portal offers CSV, ODS and RDF formats and supports multiple language packages.

### Independent job-description source

For a later external-validity check, consider a public job-posting corpus only after checking that it actually contains full descriptions and has suitable reuse terms. Some job datasets contain only links or metadata, not the description text. Never assume a dataset's title guarantees the fields needed.

## Phase 1 acceptance criteria

- [ ] Dataset source, version, access date and license recorded.
- [ ] Dataset files inspected before feature engineering.
- [ ] Canonical job schema documented.
- [ ] CSV validator passes on the selected job table.
- [ ] Duplicate IDs, missing text, empty descriptions and label distribution reported.
- [ ] No personal resumes or third-party data are committed to Git.
- [ ] No invented evaluation scores or claims.

## Methodology

### Data audit
Inspect row counts, columns, nulls, duplicate IDs, text lengths, language, label distributions, and candidate/job/match key integrity.

### Canonical representation
Normalize source columns into:
- `job_id`
- `title`
- `description`
- optional `company`, `location`, `skills`

Maintain candidate records separately and link pair labels through stable IDs.

### Leakage-aware split
Split by candidate and/or job IDs rather than randomly splitting candidate-job pairs. Random pair splitting can place the same candidate or job in both train and test and inflate evaluation. For ranking, define a query as a candidate and rank a pool of jobs, or define a job and rank candidate profiles.

### Baselines and metrics (later phases)
- Baseline: TF-IDF + cosine similarity.
- Semantic model: compact sentence-transformer, CPU inference.
- Hybrid: lexical score + semantic score + explicit skill coverage.
- Metrics: Precision@K, Recall@K, MRR and NDCG@K.
- Report latency and memory use on CPU.
- Use an ablation to measure the contribution of each component.

## What not to claim

Do not claim accuracy, fairness, improved hiring quality, or real-world effectiveness until measured on a documented and defensible evaluation set. A dataset's provided compatibility score may be a proxy label, not human-validated relevance.

---

## **Phase 1: Dataset audit and validation results**

### **1. Dataset inspection**

The selected Hugging Face dataset was downloaded and inspected locally.

| Dataset | Rows | Columns |
|---|---:|---:|
| `cv.csv` | 9,544 | 11 |
| `job.csv` | 1,167 | 10 |
| `matches.csv` | 122,560 | 13 |

### **2. Normalized datasets**

Two canonical CSV files were created while preserving the original downloaded files.

**Jobs: `data/processed/jobs_normalized.csv`**
- Rows: 1,167
- Columns: 3
- Fields: `job_id`, `title`, `description`
- Missing values: 0

**Candidates: `data/processed/candidates_normalized.csv`**
- Rows: 9,544
- Columns: 7
- Fields: `candidate_id`, `resume_text`, `clean_skills`, `education_text`, `experience_text`, `years_experience`, `inferred_category`

The original source files remain in `data/raw/`.

### **3. Match label distribution**

The match dataset contains the following label distribution:

| Label | Records |
|---|---:|
| `medium_positive` | 70,636 |
| `weak_positive` | 51,150 |
| `strong_positive` | 774 |
| **Total** | **122,560** |

**Important methodological limitation:**

All available match labels are positive categories of different strengths. The dataset does not provide verified negative examples or observed non-matches. Therefore, unlisted candidate-job pairs must not automatically be treated as negative examples.

The provided compatibility scores are dataset-generated signals, not verified hiring outcomes or human-validated relevance judgments.

### **4. Data integrity checks**

The following checks were completed:

- Required columns checked for all three datasets.
- Missing required values checked.
- Duplicate candidate and job IDs checked.
- Duplicate candidate-job pairs checked.
- Candidate and job IDs referenced by the matches dataset checked against their respective source tables.

Relationship validation results:

| Check | Result |
|---|---:|
| Total match pairs | 122,560 |
| Unknown job IDs | 0 |
| Unknown candidate IDs | 0 |
| Duplicate candidate-job pairs | 0 |

All three individual dataset validations passed, and the cross-dataset relationship validation passed.

### **5. Automated testing**

The dataset validator was extended to support jobs, candidates, and matches datasets, including relationship integrity checks.

Automated tests were executed using `pytest`.

**Result: 9/9 tests passed.**

The tests cover valid datasets, duplicate candidate-job pairs, and unknown candidate/job IDs, along with the original job validator tests.

### **6. Phase 1 completion status**

- [x] Dataset files downloaded and inspected.
- [x] Canonical job schema created.
- [x] Candidate dataset normalized.
- [x] Job dataset normalized.
- [x] Individual dataset validation implemented.
- [x] Cross-dataset relationship validation implemented.
- [x] Automated tests passed (9/9).
- [x] Original raw files preserved separately.
- [ ] Dataset provenance, exact version, access date, and redistribution permissions fully verified.
- [ ] Text-length and language analysis completed.
- [ ] ESCO version and access details recorded.

### **7. Data governance and limitations**

The raw dataset is currently retained locally and is not intended for public redistribution until its provenance and reuse permissions are verified.

The project is an educational and research-oriented retrieval prototype. It must not be presented as a validated hiring decision system.

No accuracy, fairness, or real-world hiring effectiveness claims are made at this stage.
---

## **8. Text quality analysis**

Text quality analysis was performed on the cleaned job and candidate datasets.

### Job description analysis

- Total records: 1,167
- Empty texts: 0
- Texts below 100 characters: 0
- Average characters: 3,753.43
- Median characters: 3,443
- Minimum characters: 167
- Maximum characters: 13,613
- Average words: 516.34

### Resume analysis

- Total records: 9,544
- Empty texts: 0
- Texts below 100 characters: 0
- Average characters: 1,397.70
- Median characters: 1,301
- Minimum characters: 493
- Maximum characters: 3,000
- Average words: 165.45

### Observation

All records contain non-empty cleaned text, and none fall below the exploratory threshold of 100 characters.

Job descriptions are longer on average than candidate resumes. These findings support proceeding to baseline NLP experiments, although text length alone does not establish document quality or relevance.

## 9. Initial TF-IDF Baseline Evaluation

### Model
- Algorithm: TF-IDF Vectorization
- Similarity metric: Cosine Similarity
- Maximum features: 10,000
- N-gram range: Unigrams and bigrams
- Stop words: English
- Evaluation sample: First 10 candidates
- Recommendations per candidate: Top 5

### Initial Results

| Metric | Result |
|---|---:|
| Precision@5 | 0.0800 |
| Recall@5 | 0.0322 |
| MRR | 0.2533 |

### Observations
- The TF-IDF model successfully generated ranked job recommendations.
- Four out of ten evaluated candidates had at least one recorded positive match in their top five recommendations.
- Six candidates had no recorded positive match in their top five.
- The initial results indicate limitations in lexical similarity-based matching.

### Limitations
- Evaluation covers only ten candidates and is not representative of the complete dataset.
- Weak, medium and strong positive labels are currently treated equally.
- Unlisted candidate-job pairs are unknown rather than verified negative examples.
- These metrics measure retrieval of recorded positive matches, not actual hiring outcomes.

### Next Steps
- Evaluate a larger candidate sample.
- Investigate label-strength-aware evaluation.
- Develop semantic matching using Sentence Transformers.
- Compare TF-IDF and semantic matching under the same evaluation protocol.
## 10. Expanded TF-IDF Baseline Evaluation

### Evaluation Configuration
- Model: TF-IDF
- Similarity: Cosine Similarity
- Candidate sample: 100 randomly selected candidates
- Random state: 42
- Recommendations: Top 5 jobs per candidate

### Results

| Metric | 100-candidate result |
|---|---:|
| Precision@5 | 0.0920 |
| Recall@5 | 0.0464 |
| MRR | 0.1805 |

### Comparison with Initial Evaluation

The initial 10-candidate sample produced Precision@5 of 0.0800, Recall@5 of 0.0322 and MRR of 0.2533.

The expanded sample produced different values. This is expected because the evaluated candidate sets differ.

### Key Findings
- TF-IDF provides a functional lexical matching baseline.
- Retrieval of recorded positive matches remains limited.
- Keyword-based matching may not capture semantic relationships between skills and job requirements.
- A semantic matching approach is worth investigating.

### Methodological Limitations
- The evaluation uses a random sample of 100 candidates, not the complete dataset.
- All recorded positive labels are treated equally.
- Unlisted pairs are not verified negative examples.
- Results should not be interpreted as hiring success or real-world recommendation accuracy.

### Next Experiment
Implement a Sentence Transformer-based semantic matching model and compare it with the TF-IDF baseline using the same candidate sample and evaluation protocol.
## 11. Semantic Matching Evaluation

### Model Configuration
- Model: sentence-transformers/all-MiniLM-L6-v2
- Framework: Sentence Transformers
- Device: CPU
- Job descriptions: 1,167
- Evaluation candidates: 100
- Top-K: 5
- Random state: 42

### Semantic Matching Results

| Metric | Score |
|---|---:|
| Precision@5 | 0.0720 |
| Recall@5 | 0.0307 |
| MRR | 0.1483 |

### Model Comparison

| Metric | TF-IDF | Semantic Matching |
|---|---:|---:|
| Precision@5 | 0.0920 | 0.0720 |
| Recall@5 | 0.0464 | 0.0307 |
| MRR | 0.1805 | 0.1483 |

### Observations
- Both models successfully generated job recommendations.
- TF-IDF achieved higher scores on all three retrieval metrics in this experiment.
- Semantic similarity alone did not outperform the lexical baseline.
- This motivates experimentation with hybrid matching, combining lexical and semantic similarity.

### Evaluation Limitations
- Evaluation uses 100 reproducibly sampled candidates.
- Recorded positive matches are treated as relevant.
- Unlisted candidate-job pairs are not verified negative examples.
- These metrics measure retrieval of recorded labels, not real-world hiring success.

### Next Experiment
Develop a hybrid matching model combining TF-IDF and semantic similarity, then evaluate it using the same candidate sample and metrics.

## 12. Hybrid Matching Evaluation

### Model Configuration
- TF-IDF weight: 0.5
- Semantic weight: 0.5
- Semantic model: all-MiniLM-L6-v2
- Evaluation candidates: 100
- Top-K: 5
- Random state: 42

### Results

| Metric | TF-IDF | Semantic | Hybrid |
|---|---:|---:|---:|
| Precision@5 | 0.0920 | 0.0720 | 0.0740 |
| Recall@5 | 0.0464 | 0.0307 | 0.0328 |
| MRR | 0.1805 | 0.1483 | 0.1715 |

### Observations
- The hybrid model successfully combined lexical and semantic similarity.
- Hybrid matching outperformed semantic matching on all three metrics.
- TF-IDF remained the strongest model in this experiment.
- Equal weighting did not improve performance over the TF-IDF baseline.

### Limitations
- Results are based on 100 sampled candidates.
- Recorded positive matches are treated as relevant.
- Unlisted candidate-job pairs are not verified negative examples.
- Results do not measure real-world hiring success.

### Next Experiment
Investigate different TF-IDF and semantic weighting combinations. Compare their retrieval performance while avoiding overfitting to the evaluation sample.

## 13. Hybrid Weight Optimization Experiment

### Experiment Configuration
- TF-IDF and semantic similarity combined using weighted scoring.
- Evaluation candidates: 100
- Top-K: 5
- Random state: 42
- Weight combinations tested: 80:20, 70:30, 60:40, 40:60, 20:80.

### Results

| TF-IDF Weight | Semantic Weight | Precision@5 | Recall@5 | MRR |
|---:|---:|---:|---:|---:|
| 0.8 | 0.2 | 0.0800 | 0.0368 | 0.1742 |
| 0.7 | 0.3 | 0.0820 | 0.0378 | 0.1948 |
| 0.6 | 0.4 | 0.0740 | 0.0332 | 0.1757 |
| 0.4 | 0.6 | 0.0820 | 0.0367 | 0.1618 |
| 0.2 | 0.8 | 0.0680 | 0.0314 | 0.1460 |

### Best Configuration
- TF-IDF weight: 0.7
- Semantic weight: 0.3
- Precision@5: 0.0820
- Recall@5: 0.0378
- MRR: 0.1948

### Observations
- The 70:30 combination achieved the highest MRR among tested configurations.
- The 70:30 combination also achieved the highest Recall@5.
- TF-IDF alone still achieved higher Precision@5 and Recall@5.
- Semantic similarity can contribute useful ranking information, but its optimal contribution needs further validation.

### Limitations
- Weight selection and evaluation used the same 100-candidate sample.
- Results may be affected by sample-specific tuning.
- A separate held-out evaluation is required before claiming generalization.
- Recorded positive labels do not represent verified hiring outcomes.

### Next Step
Evaluate the selected 70:30 configuration on a separate held-out candidate sample.


## 14. Held-Out Evaluation

### 14.1 Objective

A separate holdout evaluation was conducted to test whether the TF-IDF, semantic, and hybrid matching models could retrieve relevant jobs for candidates not used during the initial weight-tuning experiment.

### 14.2 Experimental Setup

- Tuning sample size: 100 candidates
- Holdout sample size: 100 candidates
- Tuning random state: 42
- Holdout random state: 123
- Number of job listings: 1,167
- Top-K recommendations: 5
- Hybrid weights: 70% TF-IDF and 30% semantic similarity
- Hardware: CPU-based execution

The original tuning candidates were excluded from the holdout sample. There were zero overlapping candidates.

### 14.3 Results

| Model | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| TF-IDF | 0.1420 | 0.0918 | 0.2543 |
| Semantic | 0.0640 | 0.0355 | 0.1687 |
| Hybrid 70:30 | 0.1200 | 0.0805 | 0.2550 |

### 14.4 Observations

1. TF-IDF achieved the highest Precision@5 and Recall@5.
2. Semantic matching performed lower than both other approaches.
3. Hybrid matching achieved the highest MRR, although its advantage over TF-IDF was very small.
4. The hybrid model did not outperform TF-IDF across all evaluation metrics.
5. Results suggest that lexical matching is an important component for this dataset.

### 14.5 Limitations

- The holdout evaluation contains only 100 candidates.
- The dataset contains positive match labels but does not provide verified negative examples.
- Unlisted candidate-job pairs must not be interpreted as confirmed mismatches.
- These metrics measure retrieval against recorded positive labels, not actual hiring outcomes.
- Further evaluation on larger and independently verified datasets is required.

### 14.6 Conclusion

The holdout experiment provides an additional evaluation of the three matching approaches on candidates excluded from the initial tuning sample.

TF-IDF achieved the strongest precision and recall, while the 70:30 hybrid model achieved a marginally higher MRR. Therefore, neither approach can yet be considered universally superior. Further experiments will investigate skill-based matching and evidence-based explanations.
## 15. Skill Intelligence and Explainability Evaluation

### 15.1 Skill Extraction Evaluation

The skill extraction component was evaluated against the available annotation-supported skill labels.

| Dataset | Evaluated Records | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Candidates | 8,847 | 0.7424 | 1.0000 | 0.8522 |
| Jobs | 985 | 0.5242 | 0.7064 | 0.6018 |

**Observations:**
1. Candidate skill extraction achieved higher precision and F1-score than job skill extraction.
2. Job descriptions presented greater extraction challenges, including incomplete skill annotations and varied terminology.
3. Candidate recall of 1.0 applies only to the annotation-supported labels used in this evaluation. It does not establish complete real-world skill coverage.

### 15.2 Job Skill Classification: Human Review

A manually reviewed sample of 178 job-skill records was used to evaluate the required, preferred and unclassified labels.

Human-reviewed label distribution:

| Label | Records |
|---|---:|
| Required | 10 |
| Preferred | 27 |
| Unclassified | 141 |
| **Total** | **178** |

**Classification results:**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Required | 1.0000 | 0.9000 | 0.9474 |
| Preferred | 0.9643 | 1.0000 | 0.9818 |
| Unclassified | 1.0000 | 1.0000 | 1.0000 |
| Macro average | 0.9881 | 0.9667 | 0.9764 |
| Weighted average | 0.9946 | 0.9944 | 0.9943 |

Overall accuracy: **0.9944 (99.44%)**

**Confusion matrix:**

| Actual \ Predicted | Required | Preferred | Unclassified |
|---|---:|---:|---:|
| Required | 9 | 1 | 0 |
| Preferred | 0 | 27 | 0 |
| Unclassified | 0 | 0 | 141 |

Only one record was misclassified: an Accounting skill labelled as Preferred by the classifier but Required by the human review.

**Limitations:**
- The evaluation covers 178 manually reviewed records, not the complete job dataset.
- The reviewed sample may not represent the distribution of all job-skill records.
- The high accuracy should not be interpreted as independently established accuracy across all job descriptions.
- Additional review on a larger and independently sampled dataset is required.

### 15.3 Explainability Consistency Audit

The explanation engine was evaluated using real candidate and job records.

| Audit measure | Result |
|---|---:|
| Candidate-job pairs evaluated | 25 |
| Total consistency checks | 125 |
| Checks passed | 125 |
| Checks failed | 0 |
| Check pass rate | 100% |

The audit checked:
1. Matching scores remained within the expected range.
2. Extracted evidence contained valid matched-skill information.
3. Job skill classification counts agreed with their corresponding lists.
4. Skill gap counts agreed with the detailed skill lists.
5. Summary percentages were consistent with the detailed skill-gap analysis.

**Interpretation:** The audit confirms that the tested explanation outputs were internally consistent. It does not establish that every explanation is semantically correct or that every extracted skill is relevant.

### 15.4 Skill Knowledge Graph

The skill knowledge graph was constructed using skill co-occurrence information from job descriptions.

| Graph measure | Result |
|---|---:|
| Total nodes | 4,945 |
| Skill nodes | 4,924 |
| Category nodes | 21 |
| Directed skill relationships | 21,378 |
| Unique skill relationships | 10,689 |

Jaccard similarity was used to measure skill co-occurrence associations.

These relationships represent observed associations in job descriptions. They do not imply that two skills are technically equivalent or interchangeable.

### 15.5 Evaluation Artifacts

The following evaluation outputs were generated:

- `data/processed/holdout_evaluation_results.csv`
- `data/processed/skill_extraction_evaluation.csv`
- `data/processed/job_skill_classification_results.csv`
- `data/processed/job_skill_classifier_evaluation.csv`
- `data/processed/job_skill_classifier_confusion_matrix.csv`
- `data/processed/explainability_audit_results.csv`
- `data/processed/manual_skill_review.csv`

### 15.6 Overall Evaluation Conclusion

The project now includes multiple evaluation layers covering job retrieval, skill extraction, required/preferred skill classification, explainability consistency and skill relationship analysis.

The results demonstrate that the individual components can be evaluated using measurable criteria. However, the current dataset has limitations, including incomplete annotations, a relatively small manually reviewed classification sample and the absence of verified hiring outcomes.

Further validation using larger, independently reviewed datasets is necessary before making claims about real-world recruitment effectiveness.

The evaluation results should therefore be interpreted as evidence of prototype performance on the available data rather than proof of production-level hiring accuracy.
