# train.py
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from utils import preprocess

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Preprocess
df["clean"] = df["text"].apply(preprocess)

# FEATURE EXTRACTION (BEST FOR LANGUAGE ID)
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2,5),
    max_features=10000
)

X = vectorizer.fit_transform(df["clean"])
y = df["language"]

# Train-test split (IMPORTANT FIX)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -------------------------
# MODEL 1: Logistic Regression
# -------------------------
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, lr_pred))

# -------------------------
# MODEL 2: SVM (BEST)
# -------------------------
svm = LinearSVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM MODEL")
print(classification_report(y_test, svm_pred))

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, svm_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------------
# SAVE MODEL
# -------------------------
pickle.dump(svm, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

# Save report
report = classification_report(y_test, svm_pred)
with open("model/report.txt", "w") as f:
    f.write(report)

print("\n✅ Model saved successfully")