# ============================================================
# SENTIMENT ANALYSIS FOR STUDENT COURSE EVALUATIONS
# CLEAN + STABLE VERSION (NO NLTK ERRORS)
# ============================================================

import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download only what we need (safe)
nltk.download("stopwords")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
DATA_PATH = r"C:\Users\AishaDeen\Desktop\BI 2\LABS\BBT4206-lab-on-NLP-c6\data\reviews.csv"
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 2. COMBINE ALL TEXT COLUMNS
# ------------------------------------------------------------
TEXT_COLUMNS = [
    'f_1_In_your_opinion_which_topics_(if_any)_should_be_added_to_the_Business_Intelligence_I_and_II_curriculum',
    'f_2_In_your_opinion_which_topics_(if_any)_should_be_removed_from_the_Business_Intelligence_I_and_II_curriculum',
    'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course',
    'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'
]

df["text"] = df[TEXT_COLUMNS].astype(str).apply(lambda row: " ".join(row.values), axis=1)
df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

# ------------------------------------------------------------
# 3. CLEAN TEXT (NO WORD_TOKENIZE!)
# ------------------------------------------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()  # avoid word_tokenize to prevent punkt errors
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# ------------------------------------------------------------
# 4. TF-IDF FEATURE EXTRACTION
# ------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. SAVE ARTIFACTS
# ------------------------------------------------------------
joblib.dump(model, "sentiment_classifier.pkl")
joblib.dump(tfidf, "sentiment_vectorizer.pkl")

print("\n=====================================================")
print("Sentiment Analysis Model Trained Successfully!")
print("Saved: sentiment_classifier.pkl, sentiment_vectorizer.pkl")
print("=====================================================\n")
