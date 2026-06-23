# Rwanda Political Sentiment Analysis

<p align="center">
  <img src="images/Flag_of_Rwanda.svg" alt="Rwanda Flag" width="80"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/HuggingFace-AfroXLMR-yellow?style=flat-square&logo=huggingface"/>
  <img src="https://img.shields.io/badge/Accuracy-77.2%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tweets-678-informational?style=flat-square"/>
</p>

<p align="center">
  An end-to-end NLP pipeline that collects, preprocesses, labels, and classifies
  sentiment in tweets about Rwandan politics — with an interactive Streamlit dashboard.
</p>

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Setup & Usage](#setup--usage)
- [Lessons Learned](#lessons-learned)
- [Author](#author)

---

## Overview

This project analyzes public sentiment on Twitter/X about Rwandan political topics using machine learning and NLP. It covers the full ML pipeline — from data collection to a deployed interactive dashboard.

**Problem:** What is the overall sentiment of English-language tweets about Rwandan politics, and which words most influence that sentiment?

**Approach:**
1. Collect tweets using keyword search (no expensive API — auth token approach via Scweet)
2. Clean and preprocess text
3. Label sentiment using VADER auto-labeling + 200 manual annotations
4. Train and compare two models: Logistic Regression baseline vs AfroXLMR transformer
5. Explain predictions using SHAP values
6. Deploy an interactive dashboard on Streamlit Cloud

---

## Demo

> 🚀 **[Live Dashboard →](https://your-streamlit-app-url.streamlit.app)**
> *(update this link after deploying to Streamlit Cloud)*

| Overview | Trends | Explainability | Live Predictor |
|---|---|---|---|
| Sentiment distribution charts | Monthly trend lines | SHAP word importance | Real-time tweet classification |

---

## Project Structure

```
rwanda_sentiment/
├── data/
│   ├── raw/                        # collected tweets (CSV)
│   └── processed/                  # cleaned, labeled datasets
├── src/
│   ├── collect_tweets.py           # Phase 1: Twitter/X scraping
│   ├── preprocess.py               # Phase 2: text cleaning
│   ├── label.py                    # Phase 3: VADER + manual labeling
│   ├── train.py                    # Phase 4: model training
│   └── explain.py                  # Phase 5: SHAP explainability
├── dashboard/
│   └── app.py                      # Phase 6: Streamlit dashboard
├── models/
│   ├── logistic_regression.pkl     # trained baseline model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   └── afroxlmr_final/            # fine-tuned transformer
├── results/
│   └── explainability/             # SHAP charts and CSV
├── images/
│   └── Flag_of_Rwanda.svg
├── requirements.txt
└── README.md
```

---

## Pipeline

```
📥 Data Collection      →    🧹 Preprocessing      →    🏷️  Labeling
   Scweet + X auth            Clean text                  VADER auto-label
   678 tweets                 Remove URLs/mentions        200 manual corrections
   9 search queries           Language detection          final_label column

        ↓
🤖 Model Training       →    🔍 Explainability     →    📊 Dashboard
   LR + TF-IDF baseline       SHAP values                Streamlit app
   AfroXLMR transformer       Top words per class        4 interactive tabs
   77.2% accuracy             Per-tweet breakdowns       Live predictor
```

---

## Results

### Model Comparison

| Model | Accuracy | F1 (weighted) | Training Time |
|---|---|---|---|
| Logistic Regression + TF-IDF | 69.1% | 0.68 | < 1 second |
| **AfroXLMR (fine-tuned)** | **77.2%** | **0.77** | ~27 minutes (CPU) |

### AfroXLMR — Classification Report

```
              precision    recall  f1-score   support

    negative       0.76      0.86      0.81        44
     neutral       0.63      0.74      0.68        23
    positive       0.81      0.70      0.75        69

    accuracy                           0.77       136
```

### Dataset

| Metric | Value |
|---|---|
| Total tweets collected | 692 |
| After preprocessing | 678 |
| Positive | 345 (50.9%) |
| Negative | 220 (32.4%) |
| Neutral | 113 (16.7%) |
| Manually reviewed | 200 |
| VADER corrections | 81 (40.5%) |

---

## Key Findings

**🔴 Negative sentiment drivers:**
`genocide`, `president paul`, `congress`, `watch`, `real` — critical language around accountability and human rights.

**🟢 Positive sentiment drivers:**
`leadership`, `thank`, `africa`, `good`, `african` — praise, pride, and regional recognition.

**⚪ Neutral sentiment drivers:**
`arrived`, `country`, `paul`, `leadership` — factual, reporting-style language.

**Notable insight:** The margin between AfroXLMR (77.2%) and Logistic Regression (69.1%) highlights that transformers better capture political nuance — but classical ML remains competitive on small datasets. Removing stopwords improved AfroXLMR by 1.5% while reducing the baseline by 5.2%, suggesting transformers extract contextual meaning beyond surface word frequency.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data collection | Scweet 5.3 |
| Data processing | pandas, numpy, langdetect |
| NLP / ML | scikit-learn, HuggingFace Transformers, AfroXLMR |
| Sentiment baseline | VADER (vaderSentiment) |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Version control | Git, GitHub |

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/thunderbolt250/rwanda-sentiment-analysis.git
cd rwanda-sentiment-analysis
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
# Phase 1 — collect tweets (requires X auth token)
python src/collect_tweets.py

# Phase 2 — preprocess
python src/preprocess.py

# Phase 3 — label
python src/label.py

# Phase 4 — train models
python src/train.py

# Phase 5 — explainability
python src/explain.py

# Phase 6 — launch dashboard
streamlit run dashboard/app.py
```

---

## Lessons Learned

**1. Excel corrupts large tweet IDs**
Large numeric IDs (tweet IDs) are silently converted to scientific notation in Excel (e.g. `1640030000000000000` → `1.64003e+18`), breaking ID-based merges. Fixed by switching to text-based matching on `clean_text`.

**2. Transformer models need careful tuning on small datasets**
AfroXLMR initially collapsed to 50.7% accuracy — always predicting the majority class. Fixed with class-weighted loss, lower learning rate (`2e-5`), smaller batch size (8), and more epochs (10).

**3. Stopword removal affects models differently**
Removing stopwords hurt Logistic Regression (−5.2%) but improved AfroXLMR (+1.5%), revealing a fundamental difference in how the two approaches use language features.

**4. Human annotation catches what VADER misses**
VADER was wrong on 81 out of 200 reviewed tweets (40.5%) — especially on political sarcasm and tweets using strong language for positive purposes (e.g. "Rwanda fought hard and won").

**5. Auth token scraping has daily rate limits**
Scweet hits X's daily limit mid-collection with a single account. Planned around this by running the collector across multiple days.

---

## Author

**Mwesigye Emmy**
MSc Information Technology Student — Carnegie Mellon University Africa

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/mwesigye-emmy-a839b3199/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/thunderbolt250)

---

*Built as part of a portfolio ML project to demonstrate end-to-end NLP skills.*
