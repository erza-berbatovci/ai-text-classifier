
import os
import re
import csv
import urllib.request
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import joblib


import matplotlib.pyplot as plt


try:
    import nltk
    from nltk.corpus import stopwords
except Exception:
    nltk = None


TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}



def download_if_missing(url: str, filename: str):
    if os.path.exists(filename):
        print(f"Found '{filename}' locally — skipping download.")
        return
    print(f"Downloading {url} -> {filename} ...")
    urllib.request.urlretrieve(url, filename)
    print("Download finished.")


def load_ag_news(train_file: str = TRAIN_FILE, test_file: str = TEST_FILE):
   
    
    download_if_missing(TRAIN_URL, train_file)
    download_if_missing(TEST_URL, test_file)

    
    train = pd.read_csv(train_file, header=None, names=["class", "title", "description"]) 
    test = pd.read_csv(test_file, header=None, names=["class", "title", "description"])  

    
    for df in (train, test):
        df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
        
        df["label"] = df["class"] - 1
        df.drop(columns=["class", "title", "description"], inplace=True)

    
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    return train, test


def ensure_nltk_stopwords():
    global nltk
    if nltk is None:
        raise RuntimeError("nltk import failed. Install nltk to use custom preprocessing: pip install nltk")
    try:
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK stopwords ...")
        nltk.download("stopwords")
        nltk.download("punkt")


def simple_preprocess(text: str, remove_stopwords: bool = True):
    """Basic cleaning: lowercase, remove non-alpha, collapse spaces, optional stopword removal."""
    s = str(text).lower()
    
    s = re.sub(r"<[^>]+>", " ", s)
   
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if remove_stopwords:
        if nltk is None:
            
            builtin_stopwords = set(
                [
                    "the",
                    "a",
                    "an",
                    "in",
                    "on",
                    "at",
                    "for",
                    "and",
                    "or",
                    "is",
                    "are",
                    "was",
                    "were",
                ]
            )
            tokens = [t for t in s.split() if t not in builtin_stopwords]
        else:
            sw = set(stopwords.words("english"))
            tokens = [t for t in s.split() if t not in sw]
        return " ".join(tokens)
    return s



def plot_class_distribution(df: pd.DataFrame, title: str = "Class distribution"):
    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def top_n_words_by_class(df: pd.DataFrame, n: int = 15):
    
    vect = CountVectorizer(stop_words="english", max_features=20000)
    X = vect.fit_transform(df["text"])
    feature_names = np.array(vect.get_feature_names_out())

    results = {}
    for label in sorted(df["label"].unique()):
        mask = (df["label"] == label).values
        sums = X[mask].sum(axis=0)
        sums = np.asarray(sums).reshape(-1)
        top_idx = np.argsort(sums)[-n:][::-1]
        results[LABEL_MAP[label]] = list(feature_names[top_idx])
    return results



def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    
    
    print("Preprocessing text (this may take a little while)...")
    
    try:
        ensure_nltk_stopwords()
    except Exception:
        print("NLTK unavailable or offline — using fallback small stopword list.")

    train_texts = train_df["text"].apply(lambda t: simple_preprocess(t, remove_stopwords=True)).tolist()
    test_texts = test_df["text"].apply(lambda t: simple_preprocess(t, remove_stopwords=True)).tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))

    nb_pipe = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])
    lr_pipe = Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000))])

    
    print("Training MultinomialNB...")
    nb_pipe.fit(train_texts, y_train)
    print("Training LogisticRegression...")
    lr_pipe.fit(train_texts, y_train)

    
    preds_nb = nb_pipe.predict(test_texts)
    preds_lr = lr_pipe.predict(test_texts)

    acc_nb = metrics.accuracy_score(y_test, preds_nb)
    acc_lr = metrics.accuracy_score(y_test, preds_lr)

    print("\nResults:")
    print(f"MultinomialNB accuracy: {acc_nb:.4f}")
    print(f"LogisticRegression accuracy: {acc_lr:.4f}")

    print("\nClassification report for LogisticRegression:\n")
    print(classification_report(y_test, preds_lr, target_names=[LABEL_MAP[i] for i in range(4)]))

    
    best_pipe = lr_pipe if acc_lr >= acc_nb else nb_pipe
    best_name = "LogisticRegression" if acc_lr >= acc_nb else "MultinomialNB"
    cm = confusion_matrix(y_test, best_pipe.predict(test_texts))

    print(f"Best model: {best_name}")

    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix ({best_name})")
    plt.colorbar()
    tick_marks = np.arange(len(LABEL_MAP))
    plt.xticks(tick_marks, [LABEL_MAP[i] for i in range(4)], rotation=45)
    plt.yticks(tick_marks, [LABEL_MAP[i] for i in range(4)])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    evaluations = {
        "accuracy_nb": acc_nb,
        "accuracy_lr": acc_lr,
        "confusion_matrix": cm,
        "best_model_name": best_name,
    }

    return nb_pipe, lr_pipe, evaluations



def make_predict_function(pipe: Pipeline):
    """Return a predict(text) function bound to a fitted pipeline. The returned function
    normalizes input and returns predicted label and class probabilities (if available).
    """

    def predict(text: str):
        cleaned = simple_preprocess(text, remove_stopwords=True)
        pred = pipe.predict([cleaned])[0]
        
        probs = None
        if hasattr(pipe, "predict_proba") or hasattr(pipe.named_steps.get("clf"), "predict_proba"):
            try:
                proba = pipe.predict_proba([cleaned])[0]
                
                classes = pipe.named_steps["clf"].classes_
                probs = {str(c): float(p) for c, p in zip(classes, proba)}
            except Exception:
                probs = None
        return {"label_index": int(pred), "label": LABEL_MAP[int(pred)], "probs": probs}

    return predict



def save_pipeline(pipe: Pipeline, filename: str):
    joblib.dump(pipe, filename)
    print(f"Saved pipeline to {filename}")


def load_pipeline(filename: str):
    return joblib.load(filename)


if __name__ == "__main__":
    print("=== AG News Text Classification —")

    train_df, test_df = load_ag_news()

    # quick EDA
    print("\n-- Samples per class (train):")
    print(train_df["label"].value_counts().sort_index().rename(index=LABEL_MAP))

    plot_class_distribution(train_df, title="AG News - Training set class distribution")

    print("\n-- Top words by class (unigrams):")
    top_words = top_n_words_by_class(train_df, n=12)
    for label, words in top_words.items():
        print(f"{label}: {', '.join(words)}")

    # Train models and evaluate
    nb_pipe, lr_pipe, evals = train_and_evaluate(train_df, test_df)

    # Save best model
    best = lr_pipe if evals["best_model_name"] == "LogisticRegression" else nb_pipe
    save_pipeline(best, "best_model.joblib")

    
    predictor = make_predict_function(best)
    examples = [
        "Apple releases new iPhone with faster chip",
        "The championship match ended with a surprising comeback",
        "Stocks plunge as company misses quarterly earnings",
        "Scientists discover a new particle at the collider",
    ]

    print("\nExample predictions:")
    for ex in examples:
        print(ex)
        print(predictor(ex))


vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

# Train Logistic Regression model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=train_iter.features["label"].names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
