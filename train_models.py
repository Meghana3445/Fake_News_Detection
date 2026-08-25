import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed_news.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset Shape:", df.shape)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(subset=["cleaned_text", "label"])

X = df["cleaned_text"]
y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))


# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF Shape:", X_train_tfidf.shape)
print("Testing TF-IDF Shape :", X_test_tfidf.shape)


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf,
    vectorizer_path
)

print("\nTF-IDF Vectorizer Saved.")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

results = []

confusion_matrices = {}


# ============================================================
# TRAIN MODELS
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    confusion_matrices[model_name] = cm

    # Save model
    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + ".pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    # Display results
    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Fake News",
                "Real News"
            ],
            zero_division=0
        )
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(results_df)
)

width = 0.2

plt.bar(
    x - 1.5 * width,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    results_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    results_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    results_df["F1-Score"],
    width,
    label="F1-Score"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=20
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.title(
    "Fake News Detection - Model Comparison"
)

plt.legend()

plt.tight_layout()

comparison_graph = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_graph,
    dpi=300
)

plt.close()


# ============================================================
# FIND BEST MODEL
# ============================================================

best_index = results_df["Accuracy"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_accuracy = results_df.loc[
    best_index,
    "Accuracy"
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(best_accuracy, 4)
)


# ============================================================
# BEST MODEL CONFUSION MATRIX
# ============================================================

best_cm = confusion_matrices[
    best_model_name
]

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    best_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Fake News",
        "Real News"
    ],
    yticklabels=[
        "Fake News",
        "Real News"
    ]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.tight_layout()

confusion_graph = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_graph,
    dpi=300
)

plt.close()


# ============================================================
# SAVE BEST MODEL INFORMATION
# ============================================================

best_model_file = (
    best_model_name.lower()
    .replace(" ", "_")
    .replace("-", "")
    + ".pkl"
)

best_model_path = os.path.join(
    MODEL_DIR,
    best_model_file
)

print("\nBest Model File:")
print(best_model_path)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved Files:")

print("1. TF-IDF Vectorizer:")
print(vectorizer_path)

print("\n2. Model Results:")
print(results_file)

print("\n3. Model Comparison Graph:")
print(comparison_graph)

print("\n4. Confusion Matrix:")
print(confusion_graph)

print("\n5. Trained Models:")
print(MODEL_DIR)

print("\n" + "=" * 70)