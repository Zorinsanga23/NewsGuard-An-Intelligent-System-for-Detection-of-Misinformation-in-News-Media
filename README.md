<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.27-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-99.7%25-27AE60?style=for-the-badge"/>

<br/><br/>

# 🛡️ NEWSGUARD
### An Intelligent System for Detection of Misinformation in News Media

*A hybrid ML + Deep Learning fake news detection system with real-time analysis, live news monitoring, and an interactive Streamlit dashboard.*

<br/>

**Master of Computer Applications — Final Year Project**  
Sharda University, Greater Noida &nbsp;|&nbsp; May 2026

**Zorinsanga** `2024181683` &nbsp;&nbsp;·&nbsp;&nbsp; **Princy Kumari** `2024152084`  
*Supervised by* **Mr. Aditya Dayal Tyagi**, Assoc. Prof., Dept. of CSA

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Models Implemented](#-models-implemented)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [App Features](#-app-features)
- [Technologies Used](#-technologies-used)
- [Limitations & Future Work](#-limitations--future-work)
- [Authors](#-authors)

---

## 🔍 Overview

The rapid proliferation of misinformation across digital media platforms poses serious risks — from political manipulation to public health harm. **NEWSGUARD** addresses this by providing an end-to-end, deployable fake news detection system that combines:

- ✅ **Traditional ML models** (Naive Bayes, Logistic Regression, SVM)
- ✅ **Deep Learning models** (CNN, LSTM)
- ✅ **A novel Hybrid Model** combining TF-IDF vectors + semantic sentence embeddings
- ✅ **Real-time web application** built with Streamlit
- ✅ **Live news analysis** via NewsAPI integration

> The hybrid model achieves **99.7% accuracy** — the highest among all implemented approaches.

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.951 | 0.95 | 0.95 | 0.95 |
| Logistic Regression | 0.990 | 0.99 | 0.99 | 0.99 |
| SVM (Linear) | 0.996 | 0.99 | 0.99 | 0.99 |
| CNN | 0.992 | 0.99 | 0.99 | 0.99 |
| LSTM | 0.978 | 0.98 | 0.98 | 0.98 |
| **Hybrid Model** ⭐ | **0.997** | **0.99** | **0.99** | **0.99** |

> Evaluated on 8,980 held-out test samples from the Kaggle Fake News Dataset.

**Hybrid Model Confusion Matrix:**

| | Predicted: Real | Predicted: Fake |
|---|---|---|
| **Actual: Real** | 4,623 ✅ | 43 ❌ |
| **Actual: Fake** | 33 ❌ | 4,281 ✅ |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      INPUT LAYER                         │
│         User Text Input  │  NewsAPI Live Feed            │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                       │
│   Tokenisation → Stopword Removal → Lemmatisation        │
└──────────────┬──────────────────────────┬───────────────┘
               ▼                          ▼
      TF-IDF Vectoriser          Sentence Embedding Model
      (50,000 dimensions)            (768 dimensions)
               └──────────────┬───────────────┘
                               ▼
                    np.hstack() Concatenation
┌─────────────────────────────────────────────────────────┐
│                      MODEL LAYER                         │
│   NB │ LR │ SVM │ CNN │ LSTM │ Hybrid Classifier        │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                         │
│        Prediction + Confidence │ Dashboard │ Live Feed  │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

### Traditional Machine Learning
| Model | Feature | Notes |
|-------|---------|-------|
| **Naive Bayes** | TF-IDF | Fast baseline; multinomial with Laplace smoothing |
| **Logistic Regression** | TF-IDF | L2 regularisation, LBFGS solver |
| **SVM** | TF-IDF | LinearSVC with Platt scaling for probability output |

### Deep Learning
| Model | Architecture | Notes |
|-------|-------------|-------|
| **CNN** | Embedding → Conv1D(128, k=5) → GlobalMaxPool → Dense(64) → Sigmoid | Captures local n-gram patterns |
| **LSTM** | Embedding → LSTM(128) → LSTM(64) → Dense(32) → Sigmoid | Captures sequential dependencies |

### Hybrid Model ⭐
```
TF-IDF Vector (50k-d)  +  Sentence Embedding (768-d)
              └──── np.hstack() ────┘
                         │
              Logistic Regression Classifier
                         │
                  Real / Fake + Confidence Score
```
Combines **statistical** (TF-IDF) and **semantic** (sentence embeddings) feature representations for superior, complementary classification.

---

## 📂 Dataset

- **Source:** [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)
- **Size:** 44,898 labelled English-language news articles
- **Split:** 80% training (35,918) / 20% test (8,980)

| Class | Count | % |
|-------|-------|---|
| Real News | 21,417 | 47.7% |
| Fake News | 23,481 | 52.3% |

The near-balanced distribution prevents class-bias in model training.

---

## 📁 Project Structure

```
NEWSGUARD/
│
├── app.py                    # Streamlit web application
│
├── models/
│   ├── hbmodel.pkl           # Hybrid model (Logistic Regression)
│   ├── hbvectorizer.pkl      # TF-IDF vectorizer (hybrid)
│   ├── embedder.pkl          # Sentence embedding model
│   ├── lr_model.pkl          # Logistic Regression model
│   ├── svm_model.pkl         # SVM model
│   ├── nb_model.pkl          # Naive Bayes model
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer (ML models)
│   ├── tokenizer.pkl         # Keras tokenizer (DL models)
│   ├── cnn_model.h5          # CNN model weights
│   └── lstm_model.h5         # LSTM model weights
│
├── data/
│   ├── True.csv              # Real news articles
│   └── Fake.csv              # Fake news articles
│
├── notebooks/
│   ├── Training_ml.ipynb           # ML model training
│   ├── Training_cnn_and_lstm.ipynb # DL model training
│   └── Fake_news_detection.ipynb   # Exploration & analysis
│
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/newsguard.git
cd newsguard
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your NewsAPI key
Get a free key at [newsapi.org](https://newsapi.org) and set it in `app.py`:
```python
API_KEY = "your_api_key_here"
```

---

## 🚀 Usage

### Run the Streamlit app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

### Run predictions programmatically
```python
import pickle
import numpy as np
import re

# Load models
hb_model    = pickle.load(open("hbmodel.pkl", "rb"))
hb_vectorizer = pickle.load(open("hbvectorizer.pkl", "rb"))
embedder    = pickle.load(open("embedder.pkl", "rb"))

def predict(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tfidf_vec = hb_vectorizer.transform([text]).toarray()
    embed_vec = embedder.encode([text])
    combined  = np.hstack((tfidf_vec, embed_vec))
    prob = hb_model.predict_proba(combined)[0]
    label = "Real" if np.argmax(prob) == 1 else "Fake"
    return label, round(max(prob) * 100, 2)

label, confidence = predict("Your news article text here...")
print(f"{label} ({confidence}%)")
```

---

## 🖥️ App Features

### 🏠 Home — Real-time Analysis
Paste any news article and get an instant prediction with confidence score.

### 📊 Dashboard — Model Comparison
Interactive charts comparing Accuracy, Precision, Recall, and F1-Score across all 6 models. Includes confusion matrix visualisations.

### 🌐 Live News — NewsAPI Integration
Fetches the latest 8 news articles and classifies each one in real time using the hybrid model.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| ML Framework | Scikit-learn 1.3 |
| DL Framework | TensorFlow / Keras 2.12 |
| NLP | NLTK, spaCy, Sentence-Transformers |
| Web App | Streamlit 1.27 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Live Data | NewsAPI (requests) |
| Serialisation | pickle, joblib |

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- English-language only
- Trained on US political news (2015–2018)
- Text-only (no image/video analysis)
- Black-box predictions (no explainability)
- Dependent on NewsAPI for live feed

**Planned Enhancements:**
- [ ] Fine-tuned **BERT / RoBERTa** integration
- [ ] **Multilingual** support via mBERT or XLM-RoBERTa
- [ ] **Multimodal** analysis (images + text)
- [ ] **Explainable AI** with SHAP / LIME / attention maps
- [ ] **Continuous learning** with active learning loop
- [ ] **Cloud deployment** (Docker + Kubernetes)

---

## 👥 Authors

| Name | Roll No. | GitHub |
|------|----------|--------|
| Zorinsanga | 2024181683 | [@zorinsanga](https://github.com/zorinsanga) |
| Princy Kumari | 2024152084 | [@princykumari](https://github.com/princykumari) |

**Supervisor:** Mr. Aditya Dayal Tyagi, Assoc. Prof., Dept. of CSA  
**Institution:** Sharda University, Greater Noida — May 2026

---

## 📄 License

This project is submitted as academic work for the Master of Computer Applications programme at Sharda University. All rights reserved by the authors.

---

<div align="center">

*Built with ❤️ for the MCA Final Year Project — Sharda University 2026*

</div>
