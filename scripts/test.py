import joblib

from src.sentiment_core import (
    load_bert_model,
    predict_tfidf,
    predict_distilbert
)


MODEL_PATH = "artifacts/tfidf_model.joblib"


# ----------------------------
# Logistic Regression Test
# ----------------------------

def run_logistic_test(input_data):
    """
    Takes input_data (string or list of strings),
    runs ONLY logistic model,
    returns logistic_results variable.
    """

    tfidf_model = joblib.load(MODEL_PATH)

    if isinstance(input_data, str):
        input_data = [input_data]

    logistic_results = []

    for text in input_data:
        logistic_results.append(predict_tfidf(text, tfidf_model))

    return logistic_results


# ----------------------------
# DistilBERT Test
# ----------------------------

def run_distilbert_test(input_data):
    """
    Takes input_data (string or list of strings),
    runs ONLY distilbert model,
    returns distilbert_results variable.
    """

    bert_model = load_bert_model()

    if isinstance(input_data, str):
        input_data = [input_data]

    distilbert_results = []

    for text in input_data:
        distilbert_results.append(predict_distilbert(text, bert_model))

    return distilbert_results


if __name__ == "__main__":
    input_data = [
        "This product is amazing and works perfectly",
        "Very bad quality, broke in a week",
        "It's okay, does the job"
    ]

    logistic_results = run_logistic_test(input_data)
    distilbert_results = run_distilbert_test(input_data)

    print("Logistic Results:", logistic_results)
    print("Distilbert Results:", distilbert_results)