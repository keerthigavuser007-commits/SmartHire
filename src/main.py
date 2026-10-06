from pathlib import Path
import joblib

from resume_parser import extract_resume_text
from recommender import recommend_jobs


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = BASE_DIR / "models" / "classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

classifier = joblib.load(CLASSIFIER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ============================================================
# RESUME CLASSIFICATION
# ============================================================

def predict_category(resume_text):

    resume_vector = vectorizer.transform([resume_text])

    prediction = classifier.predict(resume_vector)

    return prediction[0]


# ============================================================
# DISPLAY JOB RECOMMENDATIONS
# ============================================================

def display_jobs(top_jobs):

    print("\n" + "=" * 70)
    print("TOP 10 RECOMMENDED JOBS")
    print("=" * 70)

    for rank, (_, job) in enumerate(
        top_jobs.iterrows(),
        start=1
    ):

        print(f"\nRank: {rank}")

        print(
            "Job Title:",
            job.get("title", "Not available")
        )

        print(
            "Company:",
            job.get("company_name", "Not available")
        )

        print(
            "Location:",
            job.get("location", "Not available")
        )

        print(
            "Experience:",
            job.get(
                "formatted_experience_level",
                "Not available"
            )
        )

        print(
            f"Match Score: "
            f"{job['match_score']:.2f}%"
        )

        print("-" * 50)


# ============================================================
# MAIN SMART HIRE PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SMART HIRE - RESUME TO JOB MATCHING SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. GET RESUME PATH
    # --------------------------------------------------------

    resume_path = input(
        "\nEnter the path of your resume: "
    ).strip()

    resume_path = resume_path.strip('"')

    try:

        # ----------------------------------------------------
        # 2. EXTRACT RESUME TEXT
        # ----------------------------------------------------

        print("\n[1/3] Reading resume...")

        resume_text = extract_resume_text(resume_path)

        print("Resume extracted successfully!")

        print(
            "Characters extracted:",
            len(resume_text)
        )

        # ----------------------------------------------------
        # 3. CLASSIFY RESUME
        # ----------------------------------------------------

        print("\n[2/3] Predicting resume category...")

        category = predict_category(resume_text)

        print("\nPredicted Resume Category:")
        print(category)

        # ----------------------------------------------------
        # 4. RECOMMEND JOBS
        # ----------------------------------------------------

        print("\n[3/3] Finding matching jobs...")

        top_jobs = recommend_jobs(
            resume_text,
            top_n=10
        )

        # ----------------------------------------------------
        # 5. DISPLAY RESULTS
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("SMART HIRE ANALYSIS RESULT")
        print("=" * 70)

        print(
            "\nResume Category:",
            category
        )

        display_jobs(top_jobs)

        # ----------------------------------------------------
        # 6. SAVE RECOMMENDATIONS
        # ----------------------------------------------------

        output_path = (
            BASE_DIR
            / "Data"
            / "processed"
            / "recommended_jobs.csv"
        )

        top_jobs.to_csv(
            output_path,
            index=False
        )

        print("\nRecommendations saved to:")
        print(output_path)

        print("\n" + "=" * 70)
        print("SMART HIRE ANALYSIS COMPLETED")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(error)