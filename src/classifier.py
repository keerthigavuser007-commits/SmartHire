import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("====================================")
print("SMART HIRE - RESUME CLASSIFIER")
print("====================================")

print("\nLoading resume data...")

file_path = "Data/processed/resume_clean.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

# Remove rows where Resume_str or Category is missing
df = df.dropna(subset=["Resume_str", "Category"])

# Convert text and category into string
df["Resume_str"] = df["Resume_str"].astype(str)
df["Category"] = df["Category"].astype(str)

# Remove empty resume text
df = df[df["Resume_str"].str.strip() != ""]

print("Dataset after cleaning:", df.shape)


# ============================================================
# 3. INPUT AND OUTPUT
# ============================================================

X = df["Resume_str"]
y = df["Category"]

print("\nNumber of resumes:", len(X))
print("Number of categories:", y.nunique())

print("\nCategories:")
print(y.unique())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# 6. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ============================================================
# 7. PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 9. SAVE MODEL
# ============================================================

print("\n====================================")
print("SAVING MODEL")
print("====================================")

os.makedirs("models", exist_ok=True)

model_path = "models/classifier.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

joblib.dump(model, model_path)

joblib.dump(vectorizer, vectorizer_path)

print("\nModel saved successfully:")
print(model_path)

print("\nTF-IDF vectorizer saved successfully:")
print(vectorizer_path)


# ============================================================
# 10. TEST WITH ONE RESUME
# ============================================================

print("\n====================================")
print("SAMPLE PREDICTION")
print("====================================")

sample_resume = X.iloc[0]

sample_vector = vectorizer.transform([sample_resume])

sample_prediction = model.predict(sample_vector)

print("\nPredicted Resume Category:")
print(sample_prediction[0])


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n====================================")
print("CLASSIFIER COMPLETED SUCCESSFULLY")
print("====================================")