# IT2011 - Artificial Intelligence and Machine Learning
## Group Assignment: Design, Implement, and Evaluate AI/ML Solutions
**Academic Year:** Year 2, Semester 1 (2026)  
**Institution:** SLIIT Faculty of Computing  
**Group ID:** `2026-Y2-S1-MET-23`  

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline applied to an assigned real-world dataset of **46,173 movie reviews**. The system addresses the multiclass classification problem of predicting human emotion categories (`sadness`, `joy`, `anticipation`, `optimism`, `anger`, `fear`, `disgust`, `surprise`) and rating predictions.

The project is executed across two primary milestones:
1. **Progress Review I (Viva 1):** Data Preprocessing, Cleaning, Outlier Handling, and Exploratory Data Analysis (EDA).
2. **Final Evaluation (Viva 2) & Report:** Model Design, Hyperparameter Tuning, Comparative Evaluation, and Ethical AI considerations.

---

## 👥 Group Member Allocation

| Member Name | Student IT Number | Assigned Preprocessing Technique (Viva 1) | Assigned ML Model (Viva 2) |
| :--- | :--- | :--- | :--- |
| **Athapaththu A. M. P. P.** | `IT25102549` | Text Cleaning, Noise Removal & Contractions | Multinomial Naive Bayes |
| **Nishara W.A.S.** | `IT25102550` | Tokenization, Lemmatization & Stopwords | Logistic Regression |
| **Fernando B. K. H.** | `IT25102631` | Multi-Label Categorical Encoding (Genres) | Linear Support Vector Machine |
| **Abdullah H.F.** *(Lead)* | `IT25102877` | **Numerical Cleaning, Outlier Handling & Feature Scaling** | **Random Forest Classifier** |
| **Sameeha M.S.F.** | `IT25103066` | Class Imbalance Mitigation (Cost-Sensitive Weights) | Gradient Boosting (XGBoost) |
| **Silva A.M.K.N.** | `IT25103132` | Feature Extraction (TF-IDF) & Dimensionality Reduction | Deep Learning (MLP / Neural Net) |

---

## 📊 Dataset Characteristics

* **Filename:** `Movies_Reviews_modified_version1.csv`
* **Volume:** 46,173 samples, 8 columns (~130 MB raw)
* **Target Feature:** `emotion` (8 classes with extreme imbalance: `sadness` 37.5% down to `surprise` 0.12%)
* **Key Attributes:**
  * `Reviews`: Raw user review texts (English)
  * `Ratings`: Numerical user score ($1.0 - 10.0$)
  * `genres`: Multi-label serialized string list (e.g. `['Comedy', 'Drama']`)
  * `movie_name` & `Description`: Movie title and contextual synopsis

---

## 📂 Repository Structure

```text
├── README.md                                          # Project overview, dataset details, member roles
├── group_pipeline.ipynb                               # Master integrated end-to-end preprocessing pipeline
├── data/
│   ├── raw/                                           # Original assigned dataset (Movies_Reviews_modified_version1.csv)
│   └── external/                                      # External reference datasets / lexicons
├── notebooks/                                         # Individual member notebooks
│   └── IT25102877_Preprocessing_OutliersScaling.ipynb # Member 4: Outliers & Scaling notebook
└── results/
    ├── eda_visualizations/                            # High-resolution EDA charts (.png)
    ├── logs/                                          # Execution and training logs
    └── outputs/                                       # Final processed datasets and feature matrices
```

---

## 🚀 Getting Started & Execution Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed along with the core scientific computing stack:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Dataset Setup
Place the assigned `Movies_Reviews_modified_version1.csv` inside `data/raw/`:
```text
data/raw/Movies_Reviews_modified_version1.csv
```

### 3. Running Notebooks
* **Member 4 Individual Notebook:**
  Launch Jupyter and open `notebooks/IT25102877_Preprocessing_OutliersScaling.ipynb`.
* **Integrated Team Pipeline:**
  Open and execute `group_pipeline.ipynb` sequentially from top to bottom. Generated EDA figures will automatically save into `results/eda_visualizations/`.
