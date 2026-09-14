# IT2011 - Artificial Intelligence and Machine Learning
## Group Assignment: Design, Implement, and Evaluate AI/ML Solutions for a Real-World Problem
**Academic Year:** Year 2, Semester 1 (2026)  
**Faculty:** Faculty of Computing — Sri Lanka Institute of Information Technology (SLIIT)  
**Group ID:** `2026-Y2-S1-MET-23`  

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline applied to an assigned real-world dataset of **46,173 movie reviews**. The system addresses the multiclass classification problem of predicting human emotion categories (`sadness`, `joy`, `anticipation`, `optimism`, `anger`, `fear`, `disgust`, `surprise`) and rating predictions.

The project is structured into two core milestones according to the official SLIIT specification:
1. **Progress Review I (Viva 1 — 25 Marks):** Data Cleaning, Domain Preprocessing, Numerical Outlier Handling, and Exploratory Data Analysis (EDA).
2. **Final Evaluation (Viva 2 — 55 Marks) & Documentation (10 Marks):** Model Design, Hyperparameter Tuning (GridSearchCV), Cross-Validation, Comparative Evaluation across 6 distinct ML models, and Ethical AI considerations.

---

## 👥 Group Member Allocation & Milestone 1 Status

| Member Name | Student IT Number | Assigned Preprocessing Technique (Viva 1) | Assigned ML Model (Viva 2) | Notebook Link | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Athapaththu A. M. P. P.** | `IT25102549` | Text Cleaning, Noise Removal & Contraction Expansion | Multinomial Naive Bayes | [`IT25102549_...`](notebooks/IT25102549_Preprocessing_TextCleaning.ipynb) | ✅ Completed |
| **Nishara W.A.S.** | `IT25102550` | Tokenization, Lemmatization & Domain Stopwords | Logistic Regression | [`IT25102550_...`](notebooks/IT25102550_Preprocessing_Lemmatization.ipynb) | ✅ Completed |
| **Fernando B. K. H.** | `IT25102631` | Categorical Multi-Label Encoding (`genres`) | Linear Support Vector Machine | [`IT25102631_...`](notebooks/IT25102631_Preprocessing_GenreEncoding.ipynb) | ✅ Completed |
| **Abdullah H.F.** *(Lead)* | `IT25102877` | **Numerical Cleaning, Outlier Capping & Feature Scaling** | **Random Forest Classifier** | [`IT25102877_...`](notebooks/IT25102877_Preprocessing_OutliersScaling.ipynb) | ✅ Completed |
| **Sameeha M.S.F.** | `IT25103066` | Class Imbalance Mitigation (Cost-Sensitive Weights) | Gradient Boosting (XGBoost) | [`IT25103066_...`](notebooks/IT25103066_Preprocessing_ImbalanceHandling.ipynb) | ✅ Completed |
| **Silva A.M.K.N.** | `IT25103132` | Feature Extraction (TF-IDF) & Dimensionality Reduction | Deep Learning (MLP / Neural Net) | [`IT25103132_...`](notebooks/IT25103132_Preprocessing_FeatureExtraction.ipynb) | ✅ Completed |

---

## 📊 Dataset Characteristics

* **Filename:** `Movies_Reviews_modified_version1.csv`
* **Size:** 46,173 records, 8 attributes (~130 MB raw)
* **Domain:** Natural Language Processing (NLP) & Sentiment/Emotion Analysis
* **Primary Target Attribute:** `emotion` (8 multiclass categories with severe 304:1 class imbalance)

### Data Dictionary

| Column Name | Data Type | Description | Handling / Role |
| :--- | :--- | :--- | :--- |
| `Unnamed: 0` | Integer | Original row index identifier | Dropped during preprocessing |
| `movie_name` | String | Title of the film | Informational metadata |
| `Reviews` | String | Raw English user review text | Primary NLP feature (Cleaning, Lemmatization, TF-IDF) |
| `Resenhas` | String | Portuguese translation of the review text | Redundant multilingual column (excluded from NLP training) |
| `genres` | String | Serialized list of movie genres (e.g. `['Drama', 'Romance']`) | Multi-label binarized into indicator columns |
| `Description` | String | Movie storyline summary / synopsis | Contextual NLP metadata |
| `Ratings` | Float ($1.0 - 10.0$) | User numerical review score | Normalized into $[0, 1]$ using `MinMaxScaler` |
| `emotion` | String | Target emotion category | Encoded target variable with balanced class weighting |

### Target Emotion Class Distribution

| Emotion | Sample Count | Percentage | Handling Strategy |
| :--- | :---: | :---: | :--- |
| `sadness` | 17,339 | 37.55% | Dominant majority class |
| `joy` | 7,861 | 17.02% | Secondary class |
| `anticipation` | 7,336 | 15.89% | Balanced representation |
| `optimism` | 4,812 | 10.42% | Moderate representation |
| `anger` | 3,638 | 7.88% | Moderate representation |
| `fear` | 3,460 | 7.49% | Moderate representation |
| `disgust` | 1,670 | 3.62% | Minority class |
| `surprise` | 57 | **0.12%** | **Critical minority class** (requires Stratified K-Fold & class weights) |

---

## 📂 Repository Layout (SLIIT Deliverable Standard)

The repository strictly conforms to the required directory structure specified in the assignment guidelines:

```text
IT2011-AIML-Project/
├── README.md                                          # Project overview, team allocation, execution guide
├── group_pipeline.ipynb                               # End-to-end integrated master preprocessing pipeline
│
├── data/
│   ├── raw/                                           # Movies_Reviews_modified_version1.csv (as provided)
│   └── external/                                      # External reference datasets / lexicons (.gitkeep)
│
├── notebooks/                                         # Individual member notebooks (one per member 1–6)
│   ├── IT25102549_Preprocessing_TextCleaning.ipynb    # Member 1: Athapaththu A. M. P. P.
│   ├── IT25102550_Preprocessing_Lemmatization.ipynb   # Member 2: Nishara W.A.S.
│   ├── IT25102631_Preprocessing_GenreEncoding.ipynb   # Member 3: Fernando B. K. H.
│   ├── IT25102877_Preprocessing_OutliersScaling.ipynb # Member 4: Abdullah H.F.
│   ├── IT25103066_Preprocessing_ImbalanceHandling.ipynb # Member 5: Sameeha M.S.F.
│   └── IT25103132_Preprocessing_FeatureExtraction.ipynb # Member 6: Silva A.M.K.N.
│
├── results/
│   ├── eda_visualizations/                            # High-resolution plots for Viva presentation
│   │   ├── member1_text_cleaning_distributions.png
│   │   ├── member2_ngram_stopword_frequency.png
│   │   ├── member3_genre_cooccurrence_heatmap.png
│   │   ├── member4_numerical_outliers_and_ratings.png
│   │   ├── member5_class_imbalance_distribution.png
│   │   └── member6_tfidf_svd_variance.png
│   ├── logs/                                          # Execution logs (.gitkeep)
│   └── outputs/                                       # Final processed dataset and features (.gitkeep)
│
└── docs/                                              # SLIIT Assignment Specification & Rubric PDFs
    ├── Group Assignment Specification.pdf
    ├── Progress Review I - Data Preprocessing and EDA.pdf
    ├── Final Evaluation - Implementation.pdf
    └── Final Evaluation - Documentation.pdf
```

---

## 🚀 Environment Setup & Execution Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed. Install the necessary machine learning and NLP packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk jupyter
```

### 2. Dataset Setup
Ensure the assigned dataset is placed in the raw data directory:
```text
data/raw/Movies_Reviews_modified_version1.csv
```
*(Note: Because the CSV is ~135 MB, it is tracked locally and excluded from git commits via `.gitignore` to adhere to GitHub's 100 MB file limit).*

### 3. Running Individual Member Notebooks
Each member can independently run and present their notebook located in `notebooks/`:
```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/IT25102877_Preprocessing_OutliersScaling.ipynb
```

### 4. Running the Common Integrated Pipeline
To execute the complete end-to-end preprocessing flow combining all 6 techniques:
1. Open `group_pipeline.ipynb` in VS Code or Jupyter Notebook.
2. Execute all cells sequentially.
3. The final preprocessed dataset ready for Phase 2 model training will be generated in `results/outputs/processed_movie_reviews.csv`.

---

## 🌿 Git Branching & Team Collaboration

Each team member works on an isolated branch to prevent merge conflicts:
* Member 1: `git checkout member-1-IT25102549`
* Member 2: `git checkout member-2-IT25102550`
* Member 3: `git checkout member-3-IT25102631`
* Member 4: `git checkout member-4-IT25102877`
* Member 5: `git checkout member-5-IT25103066`
* Member 6: `git checkout member-6-IT25103132`
