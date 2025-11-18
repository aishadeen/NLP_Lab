import streamlit as st
import joblib
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =============================
# Load All Saved Models
# =============================

lda = joblib.load("./model/topic_model_lda.pkl")
vectorizer_topic = joblib.load("./model/topic_vectorizer.pkl")

sentiment_model = joblib.load("./model/sentiment_classifier.pkl")
tfidf = joblib.load("./model/topic_vectorizer_using_tfidf.pkl")

with open("./model/topic_labels.json", "r", encoding="utf-8") as f:
    topic_labels = json.load(f)
topic_labels = {int(k): v for k, v in topic_labels.items()}

# =============================
# Preprocessing Functions
# =============================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text_topic(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_sentiment(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    stemmed = [stemmer.stem(w) for w in filtered]
    return " ".join(stemmed)

# =============================
# Prediction Functions
# =============================

def predict_topic(comment):
    cleaned = clean_text_topic(comment)
    X = vectorizer_topic.transform([cleaned])
    topic_distribution = lda.transform(X)[0]
    topic_id = int(np.argmax(topic_distribution))

    return {
        "topic_id": topic_id,
        "topic_label": topic_labels.get(topic_id, "Unknown Topic"),
        "probability": float(topic_distribution[topic_id])
    }

def predict_sentiment(comment):
    cleaned = clean_text_sentiment(comment)
    vector = tfidf.transform([cleaned])

    pred = sentiment_model.predict(vector)[0]
    confidence = sentiment_model.predict_proba(vector).max()

    return pred, round(float(confidence), 3)

# =============================
# STREAMLIT UI
# =============================

st.title("📘 Student Course Evaluation Analyzer")
st.subheader("🔍 Topic Detection + 😊 Sentiment Analysis")

user_input = st.text_area("Enter a student's evaluation:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        topic_result = predict_topic(user_input)
        sentiment, confidence = predict_sentiment(user_input)

        st.success("Analysis Complete ✔")

        st.write("### 🧩 Predicted Topic")
        st.write(f"**Topic Label:** {topic_result['topic_label']}")
        st.write(f"**Topic ID:** {topic_result['topic_id']}")
        st.write(f"**Probability:** {topic_result['probability']:.4f}")

        st.write("### 😊 Sentiment Analysis")
        sentiment_color = {
            "positive": "green",
            "neutral": "orange",
            "negative": "red"
        }.get(sentiment, "black")

        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence * 100:.1f}%")
