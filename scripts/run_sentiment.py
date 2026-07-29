import os
import joblib

from src.sentiment_core import (
    load_data,
    create_label,
    keyword_statistics,
    train_tfidf_model,
    load_bert_model,
    compare_models,
    save_predictions
)


JSON_PATH = "data/Appliances_5.json"
MODEL_PATH = "artifacts/tfidf_model.joblib"
PREDICTIONS_PATH = "artifacts/predictions.json"

os.makedirs("artifacts", exist_ok=True)


# ----------------------------
# 1. Read JSON
# ----------------------------

df = load_data(JSON_PATH)


# ----------------------------
# 2. Create Labels
# ----------------------------

df["sentiment_label"] = df["overall"].apply(create_label)


# ----------------------------
# 3. Keyword Count
# ----------------------------

stats = keyword_statistics(df)
print("\nKeyword Stats")
print("Total Words:", stats["total_words"])
print("Unique Keywords:", stats["unique_keywords"])
print("Top Keywords:", stats["top_keywords"])


# ----------------------------
# 4. Train TF-IDF + Logistic Regression
# ----------------------------

tfidf_model, acc = train_tfidf_model(df)
joblib.dump(tfidf_model, MODEL_PATH)
print("Model saved ->", MODEL_PATH)


# ----------------------------
# 5. Load DistilBERT
# ----------------------------

bert_model = load_bert_model()


# ----------------------------
# 6. User Input Loop
# ----------------------------

all_predictions = []

while True:
    text = input("\nEnter Review (quit): ")
    if text.lower().strip() == "quit":
        break

    result = compare_models(text, tfidf_model, bert_model)
    print(result)
    all_predictions.append(result)


# ----------------------------
# 7. Save Predictions
# ----------------------------

save_predictions(all_predictions, PREDICTIONS_PATH)
print("Predictions saved ->", PREDICTIONS_PATH)