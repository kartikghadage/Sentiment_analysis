import json
import pandas as pd
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import pipeline


# ----------------------------
# 1. Read JSON
# ----------------------------

def load_data(json_path):
    records = []

    print("Reading JSON...")

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    print("Loaded reviews:", len(records))

    df = pd.DataFrame(records)
    df = df[["reviewText", "overall"]].dropna()
    df.rename(columns={"reviewText": "review_text"}, inplace=True)

    return df


# ----------------------------
# 2. Create Labels
# ----------------------------

def create_label(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    return "negative"


# ----------------------------
# 3. Keyword Count
# ----------------------------

def keyword_statistics(df):
    words = []
    for text in df["review_text"]:
        words.extend(str(text).lower().split())
    return {
        "total_words": len(words),
        "unique_keywords": len(set(words)),
        "top_keywords": Counter(words).most_common(10)
    }


# ----------------------------
# 4. Train TF-IDF + Logistic Regression
# ----------------------------

def train_tfidf_model(df):
    X = df["review_text"]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("\nTraining Logistic Regression...")
    tfidf_model.fit(X_train, y_train)

    acc = accuracy_score(y_test, tfidf_model.predict(X_test))
    print("Accuracy:", round(acc, 4))

    return tfidf_model, acc


# ----------------------------
# 5. Load DistilBERT
# ----------------------------

def load_bert_model():
    print("\nLoading DistilBERT...")

    bert_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return bert_model


# ----------------------------
# 6. Prediction Functions
# ----------------------------

def predict_tfidf(text, tfidf_model):
    probs = tfidf_model.predict_proba([text])[0]
    classes = tfidf_model.classes_
    idx = probs.argmax()
    return {
        "label": classes[idx],
        "confidence": float(probs[idx])
    }


def predict_distilbert(text, bert_model):
    result = bert_model(text)[0]
    score = float(result["score"])
    if score < 0.65:
        label = "neutral"
    elif result["label"] == "NEGATIVE":
        label = "negative"
    else:
        label = "positive"
    return {
        "label": label,
        "confidence": score
    }


def compare_models(text, tfidf_model, bert_model):
    return {
        "text": text,
        "tfidf_logreg": predict_tfidf(text, tfidf_model),
        "distilbert": predict_distilbert(text, bert_model)
    }
# ----------------------------
# 7. Save Artifacts
# ----------------------------

def save_predictions(predictions, predictions_path):
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)