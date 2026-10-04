# IT2011 - Artificial Intelligence and Machine Learning
## Group Assignment: Design, Implement, and Evaluate AI/ML Solutions for a Real-World Problem
**Academic Year:** Year 2, Semester 1 (2026)  
**Faculty:** Faculty of Computing — Sri Lanka Institute of Information Technology (SLIIT)  
**Group ID:** `2026-Y2-S1-MET-24`  

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline applied to an assigned real-world movie reviews dataset. The system addresses the multiclass classification problem of predicting human emotion categories (`sadness`, `joy`, `anticipation`, `optimism`, `anger`, `fear`, `disgust`).

Following a thorough data audit, rigorous quality control measures were implemented in `src/data_prep.py` to eliminate data leakage and label ambiguity:
1. **Text Normalization & Deduplication:** Cleans text (contractions, HTML entities, URLs, punctuation, lowercasing) before deduplication, eliminating 26,857 repeat reviews and 4 near-repeat variations.
2. **Label Noise Mitigation (`drop_conflicts`):** Dropped review texts that appeared with conflicting emotion labels across different rows.
3. **Sparse Class Handling:** Dropped the `surprise` category (only 19 reviews post-cleaning, all originating from a single film *"She's All That"*), leaving **7 cleanly separable classes** suitable for cross-validation.
4. **Stratified Grouped Splitting:** Because every movie's reviews map to a single emotion label, all validation splits and cross-validation use `make_split()` and `get_cv()` powered by **`StratifiedGroupKFold` on `movie_name`**. This simultaneously guarantees group disjointness between movies and preserves exact class balance across all 7 emotions.
5. **Leak-Free Pipelines:** All stateful feature transformations (scalers, TF-IDF, TruncatedSVD) are encapsulated in `sklearn.pipeline.Pipeline` objects fit strictly on training splits. The exported CSV file retains only **stateless, un-leaked features** for EDA and baseline reference.

The project is structured into two core milestones according to the official SLIIT specification:
1. **Progress Review I (Viva 1 — 25 Marks):** Data Cleaning, Domain Preprocessing, Numerical Outlier Handling, and Exploratory Data Analysis (EDA).
2. **Final Evaluation (Viva 2 — 55 Marks) & Documentation (10 Marks):** Model Design, Hyperparameter Tuning (GridSearchCV), Cross-Validation, Comparative Evaluation across 6 distinct ML models, and Ethical AI considerations.

---

## 👥 Group Member Allocation & Milestone 1 Status

| Member Name | Student IT Number | Assigned Preprocessing Technique (Viva 1) | Planned Phase 2 Model | Notebook Link | Preprocessing Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Athapaththu A. M. P. P.** | `IT25102549` | Text Cleaning, Noise Removal & Contraction Expansion | Multinomial Naive Bayes | [`IT25102549_...`](notebooks/IT25102549_Preprocessing_TextCleaning.ipynb) | ✅ Completed |
| **Nishara W.A.S.** | `IT25102550` | Domain-Specific Stopword Filtering *(Lemmatization Benchmarked)* | Logistic Regression | [`IT25102550_...`](notebooks/IT25102550_Preprocessing_Lemmatization.ipynb) | ✅ Completed |
| **Fernando B. K. H.** | `IT25102631` | Categorical Multi-Label Encoding (`genres`) | Linear Support Vector Machine | [`IT25102631_...`](notebooks/IT25102631_Preprocessing_GenreEncoding.ipynb) | ✅ Completed |
| **Abdullah H.F.** *(Lead)* | `IT25102877` | **Numerical Cleaning, Outlier Capping & Feature Scaling** | **Random Forest Classifier** | [`IT25102877_...`](notebooks/IT25102877_Preprocessing_OutliersScaling.ipynb) | ✅ Completed |
| **Sameeha M.S.F.** | `IT25103066` | Class Imbalance Mitigation (Cost-Sensitive Weights) | HistGradientBoosting Classifier | [`IT25103066_...`](notebooks/IT25103066_Preprocessing_ImbalanceHandling.ipynb) | ✅ Completed |
| **Silva A.M.K.N.** | `IT25103132` | Feature Extraction (TF-IDF) & Dimensionality Reduction | Deep Learning (MLP / Neural Net) | [`IT25103132_...`](notebooks/IT25103132_Preprocessing_FeatureExtraction.ipynb) | ⚠️ Needs rerun |

---

## 📊 Dataset Characteristics

* **Filename:** `Movies_Reviews_modified_version1.csv`
* **Raw Size:** 46,173 records, 8 attributes (~130 MB raw)
* **Cleaned Size:** **16,107 unique records** across **1,309 unique movies** (produced by `src.data_prep.load_clean()`)
* **Domain:** Natural Language Processing (NLP) & Sentiment/Emotion Analysis
* **Primary Target Attribute:** `emotion` (**7 multiclass categories** post-cleaning)
* **Data Splitting Strategy:** `StratifiedGroupKFold` grouped by `movie_name` (disjoint films + stratified emotion balance)

### Data Dictionary

| Column Name | Data Type | Description | Handling / Role |
| :--- | :--- | :--- | :--- |
| `movie_name` | String | Title of the film | Grouping key for `StratifiedGroupKFold` |
| `Ratings` | Float ($1.0 - 10.0$) | User numerical review score | Scaled inside `Pipeline` via `MinMaxScaler` |
| `word_count` | Integer | Length of review in words | Capped via `WordCountIQRCapper` and scaled via `RobustScaler` in `Pipeline` |
| `emotion` | String | Target emotion category | 7 classes; balanced class weighting applied during training |
| `cleaned_review` | String | Sanitized English review text | TF-IDF feature extraction inside `Pipeline` |
| `tokens_filtered` | String | Stopword-filtered tokens | Informational tokenized review text |
| `genre_*` | Binary ($0/1$) | Binarized movie genre indicators | Categorical metadata indicators |

### Target Emotion Class Distribution (Before vs. After Cleaning)

| Emotion | Raw Count | Raw Share (%) | Cleaned Count (7 Classes) | Cleaned Share (%) | Handling Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sadness` | 17,339 | 37.55% | **6,555** | 40.70% | Majority class |
| `joy` | 7,861 | 17.02% | **2,765** | 17.17% | Secondary class |
| `anticipation` | 7,336 | 15.89% | **2,166** | 13.45% | Moderate representation |
| `optimism` | 4,812 | 10.42% | **1,772** | 11.00% | Moderate representation |
| `fear` | 3,460 | 7.49% | **1,386** | 8.61% | Moderate representation |
| `anger` | 3,638 | 7.88% | **929** | 5.77% | Minority class |
| `disgust` | 1,670 | 3.62% | **534** | 3.32% | Minority class |
| `surprise` | 57 | 0.12% | **0** | *Excluded* | Dropped: 19 reviews post-clean, all from 1 movie (*"She's All That"*) |
| **Total** | **46,173** | **100.00%** | **16,107** | **100.00%** | **Cleaned & Leak-Free** |

---

## 📂 Repository Layout (SLIIT Deliverable Standard)

The repository follows the required directory structure specified in the assignment guidelines. Optional empty directories are omitted, while Phase 2 source, result, and test folders extend the documented minimum layout.

```text
IT2011-AIML-Project/
├── README.md                                          # Project overview, team allocation, execution guide
├── group_pipeline.ipynb                               # End-to-end integrated master preprocessing pipeline
│
├── src/                                               # Shared modular source code
│   ├── data_prep.py                                   # load_clean(), make_split(), get_cv(), find_repo_root()
│   ├── preprocessors.py                               # Modular scikit-learn transformers for Members 1–6
│   ├── evaluation.py                                  # Standard result schema, metrics, validation and bootstrap helpers
│   └── protocol.py                                    # Minimal shared setup/finalization API for all model owners
│
├── data/
│   └── raw/                                           # Movies_Reviews_modified_version1.csv (as provided)
│
├── notebooks/                                         # Individual member notebooks (one per member 1–6)
│   ├── IT25102549_Preprocessing_TextCleaning.ipynb    # Member 1: Athapaththu A. M. P. P.
│   ├── IT25102550_Preprocessing_Lemmatization.ipynb   # Member 2: Nishara W.A.S.
│   ├── IT25102631_Preprocessing_GenreEncoding.ipynb   # Member 3: Fernando B. K. H.
│   ├── IT25102877_Preprocessing_OutliersScaling.ipynb # Member 4: Abdullah H.F.
│   ├── IT25103066_Preprocessing_ImbalanceHandling.ipynb # Member 5: Sameeha M.S.F.
│   ├── IT25103132_Preprocessing_FeatureExtraction.ipynb # Member 6: Silva A.M.K.N.
│   ├── IT25102877_Model_DecisionTree.ipynb             # Member 4 additional baseline
│   ├── IT25102877_Model_RandomForest.ipynb             # Member 4 primary model
│   └── Phase2_Model_Comparison.ipynb                   # Role-aware six-member comparison
│
├── results/
│   ├── eda_visualizations/                            # Contains 7 figures (members 1 to 6, with Member 3 having two: member3_genre_distribution.png and member3_genre_cooccurrence_heatmap.png)
│   │   ├── member1_text_cleaning_distributions.png
│   │   ├── member2_ngram_stopword_frequency.png
│   │   ├── member3_genre_cooccurrence_heatmap.png
│   │   ├── member3_genre_distribution.png
│   │   ├── member4_numerical_outliers_and_ratings.png
│   │   ├── member5_class_imbalance_distribution.png
│   │   └── member6_tfidf_svd_variance.png
│   ├── outputs/                                       # Stateless cleaned dataset, manifest, and serializations
│       ├── processed_movie_reviews.csv                # Stateless processed dataset (16,107 rows x 26 features)
│       ├── split_movies_manifest.joblib               # Reproducible train/test movie split manifest
│       ├── train_movies.txt                           # 1,048 train movie titles
│       ├── test_movies.txt                            # 261 test movie titles
│       └── full_preprocessor.joblib                   # Serialized ColumnTransformer fitted strictly on train_df
│   └── phase2/                                        # Standardized model metrics, predictions, plots and comparison exports
│
├── scripts/                                           # Acceptance and repository verification utilities
├── tests/
│   ├── test_preprocessing.py                          # Preprocessing and leakage checks
│   ├── test_evaluation.py                             # Saved-result schema and comparison checks
│   └── test_protocol.py                               # Shared teammate protocol checks
├── requirements.txt                                   # Reproducible dependencies
└── docs/                                              # SLIIT Assignment Specification & Rubric PDFs
    ├── Group Assignment Specification.pdf
    ├── Progress Review I - Data Preprocessing and EDA.pdf
    ├── Final Evaluation - Implementation.pdf
    └── Final Evaluation - Documentation.pdf
```

> **Note on `results/outputs/`:** Contains `processed_movie_reviews.csv` (stateless features, 16,107 x 26), `full_preprocessor.joblib`, `split_movies_manifest.joblib`, `train_movies.txt`, and `test_movies.txt`.

---

## 🚀 How to Run & Environment Setup

* **Working Directory Rule:** Open each member notebook with the working directory set to the `notebooks/` folder, and open `group_pipeline.ipynb` with the working directory set to the repo root.
* **Local Cache Handling:** The notebooks create `__pycache__/` folders when run locally; they are ignored by `.gitignore` and should be deleted before zipping.

### 1. Prerequisites
Ensure Python 3.10+ is installed. Install all project dependencies:
```bash
pip install -r requirements.txt
```
* `requirements.txt` installs the latest compatible versions.
* `requirements-lock.txt` reproduces the submitted numbers exactly.
* Results of single decision trees can differ by about 0.01 macro-F1 across library versions.
*(Note: `nltk` is included in `requirements.txt` to support Member 2's standalone exploration notebook. The core group preprocessing pipeline itself in `group_pipeline.ipynb` and `src/preprocessors.py` does not require NLTK).*

### 2. Dataset Setup
Ensure the assigned dataset is placed in the raw data directory:
```text
data/raw/Movies_Reviews_modified_version1.csv
```
*(Note: The raw dataset is not committed to git (size limit); place Movies_Reviews_modified_version1.csv in data/raw/ before running anything).*

### 3. Running Individual Member Notebooks
Each member can independently run and present their notebook located in `notebooks/` (working directory set to `notebooks/`):
```bash
# Launch Jupyter Notebook from notebooks/ folder
cd notebooks
jupyter notebook IT25102877_Preprocessing_OutliersScaling.ipynb
```
Each notebook contains the original individual preprocessing implementation plus a dedicated concluding cell showcasing its technique encapsulated inside a leak-free `sklearn.pipeline.Pipeline` evaluated via stratified grouped splits.

### 4. Running the Common Integrated Pipeline
To execute the complete end-to-end preprocessing flow combining all 6 techniques:
1. Open `group_pipeline.ipynb` with working directory set to the repo root.
2. Execute all cells sequentially.
3. The final preprocessed dataset ready for Phase 2 model training will be generated in `results/outputs/processed_movie_reviews.csv`.

---

## 🔬 Phase 2: Model Training and Comparison

Phase 2 focuses on multi-class emotion classification benchmarking across the group. Each model notebook tunes its algorithm using grouped stratified cross-validation on `train_df`, performs a single held-out evaluation on `test_df`, and serializes results via `src.evaluation.save_model_results()`. The integrated comparison notebook aggregates, verifies, and compares all models without training any estimators.

### 1. Model Notebooks & Delivery Status

The group comparison ranks one primary model per member; additional models are reported as baselines.

| Model Key | Model Label | Member ID | Role | Algorithm | Notebook Link | Delivery Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| `decision_tree` | Decision Tree | `IT25102877` | Baseline | `DecisionTreeClassifier` | [`IT25102877_Model_DecisionTree.ipynb`](notebooks/IT25102877_Model_DecisionTree.ipynb) | ✅ Delivered |
| `random_forest` | Random Forest | `IT25102877` | Primary | `RandomForestClassifier` | [`IT25102877_Model_RandomForest.ipynb`](notebooks/IT25102877_Model_RandomForest.ipynb) | ✅ Delivered |
| `naive_bayes` | Multinomial Naive Bayes | `IT25102549` | Primary | `MultinomialNB` | `IT25102549_Model_NaiveBayes.ipynb` | ⏳ Pending |
| `logistic_regression` | Logistic Regression | `IT25102550` | Primary | `LogisticRegression` | `IT25102550_Model_LogisticRegression.ipynb` | ⏳ Pending |
| `linear_svc` | Linear Support Vector Machine | `IT25102631` | Primary | `LinearSVC` | `IT25102631_Model_LinearSVC.ipynb` | ⏳ Pending |
| `hist_gradient_boosting` | Gradient Boosting | `IT25103066` | Primary | `HistGradientBoostingClassifier` | `IT25103066_Model_HistGradientBoosting.ipynb` | ⏳ Pending |
| `mlp` | Multi-Layer Perceptron | `IT25103132` | Primary | `MLPClassifier` | `IT25103132_Model_MLP.ipynb` | ⏳ Pending |

**Current standardized primary delivery:** 1 of 6 (`random_forest`). The delivered `decision_tree` artifact is an additional baseline and does not increase primary-model progress.

### 2. Execution Run Order

1. **Phase 1 Pipeline (Preprocessing):** Run `group_pipeline.ipynb` to verify data cleaning and invariant checks.
2. **Phase 2 Model Training:** Run Member 4's individual model notebooks in `notebooks/`:
   * [`IT25102877_Model_DecisionTree.ipynb`](notebooks/IT25102877_Model_DecisionTree.ipynb)
   * [`IT25102877_Model_RandomForest.ipynb`](notebooks/IT25102877_Model_RandomForest.ipynb)
   Each notebook performs hyperparameter tuning, cross-validation, and held-out test evaluation, saving standardized outputs to `results/phase2/`.
3. **Phase 2 Comparison Notebook:** Run `notebooks/Phase2_Model_Comparison.ipynb`. It automatically discovers all delivered models, executes comparability checks, and generates consolidated tables and figures in `results/phase2/comparison/`.

### 3. Artifacts Saved in `results/phase2/`

* **Per-Model Serializations:**
  * `{model_key}_results.json`: Complete metadata, split fingerprints, 5-fold CV scores, test metrics, baseline benchmarks, and confusion matrices.
  * `{model_key}_test_predictions.csv`: Row-level predictions on the test set (`row_id`, `review_id`, `movie_name`, `y_true`, `y_pred`).
  * Diagnostic plots and parameter exports (`decision_tree_cv_results.csv`, `decision_tree_best_params.json`, `random_forest_cv_results.csv`, etc.).
* **Integrated Comparison Outputs (`results/phase2/comparison/`):**
  * `model_comparison_table.csv` & `model_comparison_table.md`: Consolidated performance table including CV and test scores with bootstrap 95% CIs.
  * `comparison_macro_f1.png`: Grouped bar chart comparing CV and test macro-F1 with baseline references.
  * `comparison_per_class_f1.png`: Heatmap of per-class F1 performance across all delivered models.
  * `comparison_confusion_matrices.png`: Row-normalized confusion matrix grid.
  * `comparison_cv_folds.png`: Cross-validation fold score distribution box/strip plot.
  * `comparison_summary.json`: Serialized comparison metadata, rankings, and statistical tie decisions.

### 4. How to Add Your Model to the Comparison

Python 3.10+ is required. The protocol creates the identical cleaned dataset, movie-grouped split, five shared CV folds, metrics, fingerprints, and baselines for every member. The test set is used only inside `finalize_model()` and can be evaluated only once per context.

#### Quick path

This working example uses a dummy estimator; replace it, the parameter grid, and the metadata with your model:

```python
import time
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src.protocol import SCORING, finalize_model, standard_setup
ctx = standard_setup()
pipe = Pipeline([("prep", ctx.build_preprocessor()), ("clf", DummyClassifier())])
start = time.perf_counter()
search = GridSearchCV(pipe, {"clf__strategy": ["most_frequent"]}, scoring=SCORING, refit="macro_f1", cv=ctx.cv_splits); search.fit(ctx.X_train, ctx.y_train)
finalize_model(ctx, model_key="logistic_regression", member_id="IT25102550", model_label="Quick Path Test", algorithm="DummyClassifier", preprocessing="Shared full pipeline", search=search, elapsed_seconds=time.perf_counter() - start)
```

For Keras, PyTorch, or Colab workflows, return validation predictions from one fold function and pass the single final test prediction array to the protocol:

```python
from src.protocol import cv_scores_from_folds, finalize_model, fold_scores_for, standard_setup
ctx = standard_setup()
macro, accuracy, weighted = fold_scores_for(ctx, fit_predict)
cv_scores = cv_scores_from_folds(macro, accuracy, weighted)
y_pred = predict_final_test_once()
finalize_model(ctx, model_key="mlp", member_id="IT25103132", model_label="MLP", algorithm="Keras MLP", preprocessing="Shared fold-safe preprocessing", y_pred=y_pred, cv_scores=cv_scores, n_configs=1, tuning_method="ManualSearch", elapsed_seconds=elapsed_seconds, best_params=best_params)
```

#### Advanced: manual template

To integrate manually into `Phase2_Model_Comparison.ipynb`, every notebook must still use the identical split and folds and serialize the shared result schema.

Copy the setup sections (data, split, SPLIT_INFO, CV folds and CV_FINGERPRINTS, SCORING, BASELINES) and the Step 5 test-evaluation cell (which creates test_metrics and predictions_df) from notebooks/IT25102877_Model_DecisionTree.ipynb first, so every name used in this template exists. Set elapsed_time to the number of seconds your search took.

```python
meta = {
    "model_label": "<Model Label, e.g. Logistic Regression>",
    "member_id": "<Your IT Number, e.g. IT25102550>",
    "algorithm": "<Estimator Name, e.g. LogisticRegression>",
    "preprocessing": "<Preprocessing Description, e.g. Full Pipeline (TF-IDF + Genre + WordCount + Rating)>",
    "tuning": {
        "method": "<GridSearchCV or RandomizedSearchCV>",
        "n_configs": int(len(search.cv_results_["params"])),
        "n_folds": int(N_SPLITS),
        "scoring": list(SCORING.keys()),
        "elapsed_seconds": float(elapsed_time),
        "best_params": {k: (v if v is not None else None) for k, v in search.best_params_.items()},
    }
}

json_path, csv_path = save_model_results(
    out_dir=PHASE2_DIR,
    model_key="<model_key, e.g. logistic_regression>",
    meta=meta,
    split=SPLIT_INFO,
    cv=cv_fold_scores(search),
    cv_fingerprints=CV_FINGERPRINTS,
    test=test_metrics,
    baselines=BASELINES,
    predictions_df=predictions_df,
)

loaded = load_model_results(json_path)
validate_results(loaded)
print(f"✔ Artifacts validated and saved: {json_path.name}, {csv_path.name}")
```

---

## 🌿 Git Branching & Team Collaboration

Each team member works on an isolated branch to prevent merge conflicts:
* Member 1: `git checkout member-1-IT25102549`
* Member 2: `git checkout member-2-IT25102550`
* Member 3: `git checkout member-3-IT25102631`
* Member 4: `git checkout member-4-IT25102877`
* Member 5: `git checkout member-5-IT25103066`
* Member 6: `git checkout member-6-IT25103132`
