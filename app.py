import streamlit as st
import joblib
import json
import numpy as np
import re
from pathlib import Path

# ---------------------------
# Helper: robust model loader
# ---------------------------
def try_load_model(paths):
    """Try loading first existing path from a list; return (obj, path) or (None, None)."""
    for p in paths:
        p = Path(p)
        if p.exists():
            try:
                return joblib.load(str(p)), str(p)
            except Exception as e:
                st.error(f"Failed to load model from {p}: {e}")
                return None, str(p)
    return None, None

# ---------------------------
# Load models (robust)
# ---------------------------
st.info("Loading models...")

lda, lda_path = try_load_model([
    "./model/topic_model_lda.pkl",
    "./topic_model_lda.pkl",
    "model/topic_model_lda.pkl"
])
vectorizer_topic, vec_topic_path = try_load_model([
    "./model/topic_vectorizer.pkl",
    "./topic_vectorizer.pkl",
    "model/topic_vectorizer.pkl"
])

# sentiment model
sentiment_model, sent_model_path = try_load_model([
    "./model/sentiment_classifier.pkl",
    "./sentiment_classifier.pkl",
    "model/sentiment_classifier.pkl"
])

# try multiple possible vectorizer filenames (historical mismatch)
tfidf, tfidf_path = try_load_model([
    "./model/topic_vectorizer_using_tfidf.pkl",   # original app used this name
    "./model/sentiment_vectorizer.pkl",
    "./model/topic_vectorizer.pkl",
    "./sentiment_vectorizer.pkl",
    "./topic_vectorizer_using_tfidf.pkl",
    "./topic_vectorizer.pkl"
])

# topic labels
topic_labels = {}
labels_path = Path("./model/topic_labels.json")
if labels_path.exists():
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            topic_labels = json.load(f)
            topic_labels = {int(k): v for k, v in topic_labels.items()}
    except Exception as e:
        st.error(f"Failed to load topic_labels.json: {e}")
else:
    # fallback: auto-generate simple labels if LDA is present
    if lda is not None:
        topic_labels = {i: f"Topic {i}" for i in range(getattr(lda, "n_components", 5))}
    else:
        topic_labels = {}

# If any critical artifact missing -> show error and stop
missing = []
if lda is None: missing.append("LDA model (topic_model_lda.pkl)")
if vectorizer_topic is None: missing.append("Topic vectorizer (topic_vectorizer.pkl)")
if sentiment_model is None: missing.append("Sentiment classifier (sentiment_classifier.pkl)")
if tfidf is None: missing.append("TF-IDF vectorizer (sentiment vectorizer)")

if missing:
    st.error("One or more model artifacts are missing. Please ensure the following files exist in `./model` or project root:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

st.success("Models loaded successfully.")

# ---------------------------
# Preprocessing (no NLTK stopwords/punkt use)
# ---------------------------

# Internal stopword list (keeps app self-contained; extend as needed)
STOP_WORDS = {
    "the","and","is","in","of","for","to","on","with","was","it","as","this","that","are","be","by","an","or","at","from","but",
    "we","they","you","i","a","about","into","more","so","can","if","when","what","how","which","their","there","my","our",
    "course","lecturer","lectures","class","classes","would","could","also","not","very"
}

# Try to use PorterStemmer if nltk is installed; otherwise, use identity stemmer
try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def _stem_word(w): return stemmer.stem(w)
except Exception:
    def _stem_word(w): return w  # no-op stemmer

def clean_text_topic(text: str) -> str:
    """Light cleaning for topic modeling (lowercase, remove non-letters, collapse spaces)."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_sentiment(text: str) -> str:
    """Cleaning for sentiment pipeline (no external tokenizers)."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = text.split()  # simple split avoids punkt/tokenizer requirement
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [_stem_word(t) for t in tokens]
    return " ".join(tokens)

# ---------------------------
# Prediction functions
# ---------------------------
def predict_topic(comment: str):
    cleaned = clean_text_topic(comment)
    X = vectorizer_topic.transform([cleaned])
    topic_distribution = lda.transform(X)[0]
    topic_id = int(np.argmax(topic_distribution))
    return {
        "topic_id": topic_id,
        "topic_label": topic_labels.get(topic_id, f"Topic {topic_id}"),
        "probability": float(topic_distribution[topic_id])
    }

def predict_sentiment(comment: str):
    cleaned = clean_text_sentiment(comment)
    vector = tfidf.transform([cleaned])
    pred = sentiment_model.predict(vector)[0]
    # some classifiers may not implement predict_proba (e.g., SVC without prob)
    try:
        confidence = float(sentiment_model.predict_proba(vector).max())
    except Exception:
        confidence = 0.0
    return pred, round(confidence, 3)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📘 Student Course Evaluation Analyzer")
st.markdown(
    "Enter one student's textual evaluation below. The app returns the **predicted topic** and the **sentiment**."
)

with st.form(key="analysis_form"):
    user_input = st.text_area("Student evaluation (one entry):", height=180)
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not user_input or user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                topic_result = predict_topic(user_input)
                sentiment_label, confidence = predict_sentiment(user_input)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                raise

        st.success("Analysis complete ✔")

        st.subheader("🧩 Predicted Topic")
        st.write(f"**Topic Label:** {topic_result['topic_label']}")
        st.write(f"**Topic ID:** {topic_result['topic_id']}")
        st.write(f"**Probability:** {topic_result['probability']:.4f}")

        st.subheader("😊 Sentiment Analysis")
        sentiment_color = {"positive": "green", "neutral": "orange", "negative": "red"}.get(sentiment_label, "black")
        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence * 100:.1f}%")

