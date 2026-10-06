import pandas as pd

# ==========================================
# 1. LOAD RESUME DATASET
# ==========================================

resume_path = "Data/raw/resume.csv"

print("\n========== RESUME DATASET ==========")

try:
    resume = pd.read_csv(resume_path)

    print("\nRows and Columns:")
    print(resume.shape)

    print("\nColumn Names:")
    print(resume.columns.tolist())

    print("\nData Types:")
    print(resume.dtypes)

    print("\nMissing Values:")
    print(resume.isnull().sum())

    print("\nDuplicate Rows:")
    print(resume.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(resume.head())

except FileNotFoundError:
    print("ERROR: resume.csv not found.")
    print("Check: Data/raw/resume.csv")

except pd.errors.EmptyDataError:
    print("ERROR: resume.csv is empty.")

except Exception as e:
    print("ERROR:", e)


# ==========================================
# 2. LOAD JOB DATASET
# ==========================================

job_path = "Data/raw/indian_job.csv"

print("\n\n========== JOB DATASET ==========")

try:
    jobs = pd.read_csv(job_path)

    print("\nRows and Columns:")
    print(jobs.shape)

    print("\nColumn Names:")
    print(jobs.columns.tolist())

    print("\nData Types:")
    print(jobs.dtypes)

    print("\nMissing Values:")
    print(jobs.isnull().sum())

    print("\nDuplicate Rows:")
    print(jobs.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(jobs.head())

except FileNotFoundError:
    print("ERROR: indian_job.csv not found.")
    print("Check: Data/raw/indian_job.csv")

except pd.errors.EmptyDataError:
    print("ERROR: indian_job.csv is empty.")

except Exception as e:
    print("ERROR:", e)