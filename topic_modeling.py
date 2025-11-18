# ============================================================
# TOPIC MODELING FOR STUDENT COURSE EVALUATIONS
# ============================================================

import pandas as pd
import re
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
DATA_PATH = r"C:\Users\AishaDeen\Desktop\BI 2\LABS\BBT4206-lab-on-NLP-c6\data\reviews.csv"

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 2. COMBINE ALL TEXT FIELDS
# ------------------------------------------------------------
TEXT_COLUMNS = [
    'f_1_In_your_opinion_which_topics_(if_any)_should_be_added_to_the_Business_Intelligence_I_and_II_curriculum',
    'f_2_In_your_opinion_which_topics_(if_any)_should_be_removed_from_the_Business_Intelligence_I_and_II_curriculum',
    'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course',
    'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'
]

df["text"] = df[TEXT_COLUMNS].astype(str).apply(lambda row: " ".join(row.values), axis=1)
df = df.dropna(subset=["text"]).reset_index(drop=True)

# ------------------------------------------------------------
# 3. CLEAN TEXT
# ------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ------------------------------------------------------------
# 4. DOCUMENT-TERM MATRIX
# ------------------------------------------------------------
vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=2,
    stop_words="english",
    max_features=2000
)

dtm = vectorizer.fit_transform(df["clean_text"])

# ------------------------------------------------------------
# 5. TRAIN LDA MODEL
# ------------------------------------------------------------
lda = LatentDirichletAllocation(
    n_components=5,   # Number of topics
    random_state=42
)

lda.fit(dtm)

# ------------------------------------------------------------
# 6. SAVE ARTIFACTS
# ------------------------------------------------------------
joblib.dump(lda, "topic_model_lda.pkl")
joblib.dump(vectorizer, "topic_vectorizer.pkl")

topic_labels = {i: f"Topic {i}" for i in range(5)}

with open("topic_labels.json", "w") as f:
    json.dump(topic_labels, f, indent=4)

print("\n=====================================================")
print("Topic Modeling Completed Successfully!")
print("Saved: topic_model_lda.pkl, topic_vectorizer.pkl, topic_labels.json")
print("=====================================================\n")
