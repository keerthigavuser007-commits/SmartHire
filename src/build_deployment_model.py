import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = Path("Data/deployment/jobs_deploy.csv")
MODEL_DIR = Path("models/deployment")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading deployment jobs...")

jobs = pd.read_csv(DATA_FILE)

jobs["job_text"] = jobs["job_text"].fillna("").astype(str)

jobs = jobs[jobs["job_text"].str.strip() != ""]

print("Jobs used:", len(jobs))

print("Creating TF-IDF model...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

job_vectors = vectorizer.fit_transform(jobs["job_text"])

print("TF-IDF shape:", job_vectors.shape)

joblib.dump(
    vectorizer,
    MODEL_DIR / "job_tfidf_vectorizer.pkl"
)

joblib.dump(
    job_vectors,
    MODEL_DIR / "job_tfidf_vectors.pkl"
)

jobs.to_csv(
    MODEL_DIR / "jobs_deploy.csv",
    index=False
)

print()
print("DEPLOYMENT MODEL CREATED SUCCESSFULLY")
print("Saved to:", MODEL_DIR)