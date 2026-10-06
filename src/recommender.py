import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "models" / "deployment" / "jobs_deploy.csv"
VECTORIZER_FILE = BASE_DIR / "models" / "deployment" / "job_tfidf_vectorizer.pkl"
VECTORS_FILE = BASE_DIR / "models" / "deployment" / "job_tfidf_vectors.pkl"


def load_job_model():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Deployment job dataset not found: {DATA_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Deployment vectorizer not found: {VECTORIZER_FILE}"
        )

    if not VECTORS_FILE.exists():
        raise FileNotFoundError(
            f"Deployment TF-IDF vectors not found: {VECTORS_FILE}"
        )

    jobs = pd.read_csv(DATA_FILE)

    vectorizer = joblib.load(VECTORIZER_FILE)

    job_vectors = joblib.load(VECTORS_FILE)

    return jobs, vectorizer, job_vectors


def recommend_jobs(resume_text, top_n=10):

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    jobs, vectorizer, job_vectors = load_job_model()

    resume_vector = vectorizer.transform([resume_text])

    similarity_scores = cosine_similarity(
        resume_vector,
        job_vectors
    ).flatten()

    results = jobs.copy()

    results["match_score"] = similarity_scores * 100

    results = results.sort_values(
        by="match_score",
        ascending=False
    )

    return results.head(top_n).copy()


if __name__ == "__main__":

    print("SMART HIRE - DEPLOYMENT JOB RECOMMENDER")

    sample_resume = """
    Python machine learning data science pandas numpy
    scikit learn SQL artificial intelligence
    """

    results = recommend_jobs(sample_resume, top_n=10)

    print("\nTOP 10 RECOMMENDED JOBS\n")

    for i, (_, job) in enumerate(results.iterrows(), start=1):

        print(
            f"{i}. {job.get('title', 'Unknown')} | "
            f"{job.get('company_name', 'Unknown')} | "
            f"{job.get('location', 'Unknown')} | "
            f"{job['match_score']:.2f}%"
        )