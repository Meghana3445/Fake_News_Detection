import os
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

tfidf = joblib.load(
    vectorizer_path
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

models = {}

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
    "Random Forest": "random_forest.pkl"
}


for model_name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if os.path.exists(model_path):

        models[model_name] = joblib.load(
            model_path
        )


# ============================================================
# SELECT BEST MODEL
# ============================================================

results_file = os.path.join(
    BASE_DIR,
    "outputs",
    "model_results.csv"
)

import pandas as pd

results = pd.read_csv(
    results_file
)

best_model_name = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

model = models[best_model_name]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_news(news_text):

    # Convert text to TF-IDF
    text_vector = tfidf.transform(
        [news_text]
    )

    # Make prediction
    prediction = model.predict(
        text_vector
    )[0]

    # Convert numerical result
    if prediction == 0:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"

    return result


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 70)
print("             FAKE NEWS DETECTION SYSTEM")
print("=" * 70)

print("\nBest Model Selected:")
print(best_model_name)

print("\nEnter a news article below.")
print("Type 'exit' to close the program.")

while True:

    print("\n" + "-" * 70)

    news = input(
        "Enter News Article: "
    )

    if news.lower() == "exit":
        print("\nProgram closed.")
        break

    if not news.strip():

        print(
            "Please enter some news text."
        )

        continue

    result = predict_news(
        news
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        "Prediction:",
        result
    )

    print(
        "Model:",
        best_model_name
    )

    print("=" * 70)