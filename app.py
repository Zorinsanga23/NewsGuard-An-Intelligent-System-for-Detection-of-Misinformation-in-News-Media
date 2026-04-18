import streamlit as st
import pickle
import numpy as np
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():
    hb_model = pickle.load(open("hbmodel.pkl", "rb"))
    hb_vectorizer = pickle.load(open("hbvectorizer.pkl", "rb"))
    embedder = pickle.load(open("embedder.pkl", "rb"))

    cnn_model = load_model("cnn_model.h5")
    lstm_model = load_model("lstm_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

    return hb_model, hb_vectorizer, embedder, cnn_model, lstm_model, tokenizer

hb_model, hb_vectorizer, embedder, cnn_model, lstm_model, tokenizer = load_models()

API_KEY = "bf8beca48b224dc083c0acb7d1c0d9bb"

st.set_page_config(page_title="Fake News Detection", layout="wide")

# ---------------- STYLE ---------------- #

st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f4f6f8;
    margin-bottom: 15px;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# 🔹 Hybrid Prediction ONLY
def analyze_hybrid(text):
    cleaned = clean_text(text)

    tfidf_vec = hb_vectorizer.transform([cleaned]).toarray()
    embed_vec = embedder.encode([cleaned])

    final_vec = np.hstack((tfidf_vec, embed_vec))

    prob = hb_model.predict_proba(final_vec)[0]
    pred = np.argmax(prob)

    return ("Real", max(prob)*100) if pred == 1 else ("Fake", max(prob)*100)

# 🔹 News API
def get_live_news():
    url = f"https://newsapi.org/v2/everything?q=news&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    data = response.json()

    if data.get("status") != "ok":
        raise Exception(data.get("message", "API error"))

    return data.get("articles", [])

# ---------------- SIDEBAR ---------------- #

menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Dashboard", "🌐 Live News"])

# ---------------- HOME ---------------- #

if menu == "🏠 Home":

    st.title("📰 Fake News Detection System")

    text = st.text_area("Enter News Text")

    if st.button("Analyze"):

        label, confidence = analyze_hybrid(text)

        if label == "Real":
            st.success(f"🟢 REAL NEWS ({confidence:.2f}%)")
        else:
            st.error(f"🔴 FAKE NEWS ({confidence:.2f}%)")

        st.progress(int(confidence))

# ---------------- DASHBOARD (STATIC BUT FULL) ---------------- #

elif menu == "📊 Dashboard":

    st.title("📊 Model Comparison Dashboard")

    # 🔹 Static Metrics
    df = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "SVM", "CNN", "LSTM", "Hybrid"],
        "Accuracy": [0.90, 0.94, 0.95, 0.94, 0.95, 0.96],
        "Precision": [0.89, 0.93, 0.94, 0.93, 0.94, 0.96],
        "Recall": [0.88, 0.92, 0.94, 0.93, 0.94, 0.95],
        "F1 Score": [0.88, 0.92, 0.94, 0.93, 0.94, 0.95],
        "Speed": ["Very Fast", "Fast", "Medium", "Slow", "Very Slow", "Medium"]
    })

    # 🔹 Best Model
    st.success("🏆 Best Model: Hybrid (Proposed Model) ⭐")

    # 🔹 Table
    st.markdown("### 📋 Metrics Table")
    st.dataframe(df.style.highlight_max(axis=0))

    # 🔹 Accuracy Chart
    st.markdown("### 📊 Accuracy Comparison")
    st.bar_chart(df.set_index("Model")["Accuracy"])

    # 🔹 Multi-Metric Chart
    st.markdown("### 📊 Multi-Metric Comparison")
    st.bar_chart(df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])

    # 🔹 Speed Comparison
    st.markdown("### ⚡ Speed Comparison")
    for i, row in df.iterrows():
        st.write(f"{row['Model']} → {row['Speed']}")

    # 🔹 Confusion Matrix (Example)
    st.markdown("### 🔍 Confusion Matrix (Logistic Regression)")

    cm = [[4200, 150],
          [200, 4300]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------- LIVE NEWS ---------------- #

elif menu == "🌐 Live News":

    st.title("🌐 Live News (Hybrid Model Only)")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        articles = get_live_news()

        if not articles:
            st.warning("No news available")

        for article in articles[:8]:

            title = article.get("title")

            if not title:
                continue

            label, confidence = analyze_hybrid(title)

            color = "green" if label == "Real" else "red"

            st.markdown(f"""
            <div class="card">
                <b style="color:{color}">
                    {label} ({confidence:.2f}%)
                </b><br>
                {title}
            </div>
            """, unsafe_allow_html=True)

            # Optional: Description
            description = article.get("description")

            if description:
                with st.expander("More Details"):
                    st.write(description)

                    l2, c2 = analyze_hybrid(description)
                    st.write(f"{l2} ({c2:.2f}%)")

    except Exception as e:
        st.error(f"⚠️ {e}")