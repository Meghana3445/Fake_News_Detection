import tkinter as tk
from tkinter import messagebox
import os
import joblib
import pandas as pd
import re
import string


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

results = pd.read_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_results.csv"
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_index = results["Accuracy"].idxmax()

best_model_name = results.loc[
    best_index,
    "Model"
]

best_accuracy = results.loc[
    best_index,
    "Accuracy"
]


# ============================================================
# MODEL FILES
# ============================================================

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# LOAD BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    model_files[best_model_name]
)

model = joblib.load(model_path)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"\[.*?\]",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICTION
# ============================================================

def predict_news():

    news_text = text_box.get(
        "1.0",
        tk.END
    ).strip()

    if not news_text:

        messagebox.showwarning(
            "Input Required",
            "Please enter a news article."
        )

        return

    if len(news_text) < 20:

        messagebox.showwarning(
            "Text Too Short",
            "Please enter a longer news article."
        )

        return

    cleaned_text = clean_text(
        news_text
    )

    vector = tfidf.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        vector
    )[0]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if prediction == 0:

        result = "FAKE NEWS"

    else:

        result = "REAL NEWS"


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = max(probabilities) * 100

    elif hasattr(model, "decision_function"):

        decision = model.decision_function(
            vector
        )

        if hasattr(decision, "__len__"):

            score = abs(float(decision[0]))

        else:

            score = abs(float(decision))

        # Convert decision score to a confidence-like
        # indicator for display purposes.
        confidence = (
            50 + (50 * (score / (1 + score)))
        )

    else:

        confidence = None


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    if result == "FAKE NEWS":

        result_label.config(
            text="⚠ FAKE NEWS",
            fg="#C62828"
        )

        status_label.config(
            text="This article is classified as likely FAKE.",
            fg="#C62828"
        )

    else:

        result_label.config(
            text="✓ REAL NEWS",
            fg="#2E7D32"
        )

        status_label.config(
            text="This article is classified as likely REAL.",
            fg="#2E7D32"
        )


    model_label.config(
        text=f"Model Used: {best_model_name}"
    )

    accuracy_label.config(
        text=f"Model Accuracy: {best_accuracy * 100:.2f}%"
    )

    if confidence is not None:

        confidence_label.config(
            text=f"Prediction Confidence: {confidence:.2f}%"
        )

    else:

        confidence_label.config(
            text="Prediction Confidence: Not Available"
        )


# ============================================================
# CLEAR
# ============================================================

def clear_text():

    text_box.delete(
        "1.0",
        tk.END
    )

    result_label.config(
        text="Prediction Result",
        fg="#333333"
    )

    status_label.config(
        text="Enter a news article and click Predict.",
        fg="#555555"
    )

    confidence_label.config(
        text="Prediction Confidence: --"
    )


# ============================================================
# CHARACTER COUNTER
# ============================================================

def update_counter(event=None):

    text = text_box.get(
        "1.0",
        tk.END
    ).strip()

    character_label.config(
        text=f"Characters: {len(text)}"
    )


# ============================================================
# EXIT
# ============================================================

def exit_program():

    answer = messagebox.askyesno(
        "Exit",
        "Are you sure you want to exit?"
    )

    if answer:

        root.destroy()


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Fake News Detection System"
)

root.geometry(
    "950x760"
)

root.resizable(
    False,
    False
)

root.configure(
    bg="#F4F6F8"
)


# ============================================================
# HEADER
# ============================================================

header_frame = tk.Frame(
    root,
    bg="#263238",
    height=100
)

header_frame.pack(
    fill="x"
)


title_label = tk.Label(
    header_frame,
    text="FAKE NEWS DETECTION SYSTEM",
    font=(
        "Arial",
        25,
        "bold"
    ),
    fg="white",
    bg="#263238"
)

title_label.pack(
    pady=(20, 3)
)


subtitle_label = tk.Label(
    header_frame,
    text="Machine Learning • Natural Language Processing",
    font=(
        "Arial",
        12
    ),
    fg="#CFD8DC",
    bg="#263238"
)

subtitle_label.pack()


# ============================================================
# MAIN CONTENT
# ============================================================

content_frame = tk.Frame(
    root,
    bg="#F4F6F8"
)

content_frame.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=25
)


# ============================================================
# INPUT LABEL
# ============================================================

input_label = tk.Label(
    content_frame,
    text="Enter News Article",
    font=(
        "Arial",
        15,
        "bold"
    ),
    bg="#F4F6F8",
    fg="#263238"
)

input_label.pack(
    anchor="w"
)


# ============================================================
# TEXT BOX FRAME
# ============================================================

text_frame = tk.Frame(
    content_frame,
    bg="white",
    bd=1,
    relief="solid"
)

text_frame.pack(
    fill="x",
    pady=(8, 5)
)


text_box = tk.Text(
    text_frame,
    height=14,
    width=100,
    font=(
        "Arial",
        11
    ),
    wrap=tk.WORD,
    bd=0,
    padx=12,
    pady=12
)

text_box.pack(
    fill="both",
    expand=True
)

text_box.bind(
    "<KeyRelease>",
    update_counter
)


# ============================================================
# CHARACTER COUNTER
# ============================================================

character_label = tk.Label(
    content_frame,
    text="Characters: 0",
    font=(
        "Arial",
        9
    ),
    bg="#F4F6F8",
    fg="#777777"
)

character_label.pack(
    anchor="e"
)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    content_frame,
    bg="#F4F6F8"
)

button_frame.pack(
    pady=15
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = tk.Button(
    button_frame,
    text="PREDICT NEWS",
    command=predict_news,
    font=(
        "Arial",
        12,
        "bold"
    ),
    bg="#1565C0",
    fg="white",
    activebackground="#0D47A1",
    activeforeground="white",
    width=20,
    height=2,
    cursor="hand2"
)

predict_button.grid(
    row=0,
    column=0,
    padx=8
)


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_text,
    font=(
        "Arial",
        12,
        "bold"
    ),
    bg="#757575",
    fg="white",
    activebackground="#424242",
    activeforeground="white",
    width=14,
    height=2,
    cursor="hand2"
)

clear_button.grid(
    row=0,
    column=1,
    padx=8
)


# ============================================================
# RESULT FRAME
# ============================================================

result_frame = tk.Frame(
    content_frame,
    bg="white",
    bd=1,
    relief="solid"
)

result_frame.pack(
    fill="x",
    pady=10
)


result_title = tk.Label(
    result_frame,
    text="PREDICTION RESULT",
    font=(
        "Arial",
        12,
        "bold"
    ),
    bg="white",
    fg="#555555"
)

result_title.pack(
    pady=(15, 5)
)


result_label = tk.Label(
    result_frame,
    text="Prediction Result",
    font=(
        "Arial",
        25,
        "bold"
    ),
    bg="white",
    fg="#333333"
)

result_label.pack(
    pady=5
)


status_label = tk.Label(
    result_frame,
    text="Enter a news article and click Predict.",
    font=(
        "Arial",
        11
    ),
    bg="white",
    fg="#555555"
)

status_label.pack(
    pady=5
)


model_label = tk.Label(
    result_frame,
    text=f"Model Used: {best_model_name}",
    font=(
        "Arial",
        10
    ),
    bg="white",
    fg="#555555"
)

model_label.pack(
    pady=2
)


accuracy_label = tk.Label(
    result_frame,
    text=f"Model Accuracy: {best_accuracy * 100:.2f}%",
    font=(
        "Arial",
        10
    ),
    bg="white",
    fg="#555555"
)

accuracy_label.pack(
    pady=2
)


confidence_label = tk.Label(
    result_frame,
    text="Prediction Confidence: --",
    font=(
        "Arial",
        10,
        "bold"
    ),
    bg="white",
    fg="#333333"
)

confidence_label.pack(
    pady=(2, 15)
)


# ============================================================
# DISCLAIMER
# ============================================================

disclaimer = tk.Label(
    content_frame,
    text=(
        "Note: This system is an educational machine-learning "
        "classifier and does not independently verify factual claims."
    ),
    font=(
        "Arial",
        9
    ),
    bg="#F4F6F8",
    fg="#777777"
)

disclaimer.pack(
    pady=8
)


# ============================================================
# EXIT BUTTON
# ============================================================

exit_button = tk.Button(
    content_frame,
    text="EXIT",
    command=exit_program,
    font=(
        "Arial",
        11,
        "bold"
    ),
    bg="#C62828",
    fg="white",
    activebackground="#8E0000",
    activeforeground="white",
    width=12,
    height=2,
    cursor="hand2"
)

exit_button.pack(
    pady=5
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()