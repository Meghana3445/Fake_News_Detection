import pandas as pd
import os
import re
import string


# ============================================================
# FAKE NEWS DETECTION - DATA PREPROCESSING
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_FILE = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_FILE = os.path.join(BASE_DIR, "dataset", "True.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_news.csv")


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove text inside square brackets
    text = re.sub(r"\[.*?\]", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - DATA PREPROCESSING")
print("=" * 70)

print("\nLoading datasets...")

fake_df = pd.read_csv(FAKE_FILE)
true_df = pd.read_csv(TRUE_FILE)

print("\nFake News Records :", len(fake_df))
print("Real News Records :", len(true_df))


# ============================================================
# ADD LABELS
# ============================================================

fake_df["label"] = 0
true_df["label"] = 1


# ============================================================
# COMBINE DATASETS
# ============================================================

df = pd.concat([fake_df, true_df], ignore_index=True)

print("\nTotal Records :", len(df))


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["title", "text"])

after = len(df)

print("Duplicate Records Removed :", before - after)


# ============================================================
# COMBINE TITLE AND ARTICLE TEXT
# ============================================================

df["news"] = df["title"] + " " + df["text"]


# ============================================================
# CLEAN NEWS TEXT
# ============================================================

print("\nCleaning news text...")

df["cleaned_text"] = df["news"].apply(clean_text)


# ============================================================
# REMOVE EMPTY RECORDS
# ============================================================

df = df[df["cleaned_text"].str.strip() != ""]


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

processed_df = df[["cleaned_text", "label"]].copy()


# ============================================================
# SHUFFLE DATA
# ============================================================

processed_df = processed_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE PROCESSED DATASET
# ============================================================

processed_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal Dataset Shape:", processed_df.shape)

print("\nLabel Distribution:")

print(
    processed_df["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel Meaning:")
print("0 = Fake News")
print("1 = Real News")

print("\nProcessed Dataset Saved To:")
print(OUTPUT_FILE)

print("\nSample Data:")
print(processed_df.head())

print("\n" + "=" * 70)