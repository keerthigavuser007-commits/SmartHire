import pandas as pd
import os
import re


# ==========================================
# CREATE PROCESSED FOLDER
# ==========================================

os.makedirs("Data/processed", exist_ok=True)


# ==========================================
# 1. LOAD RESUME DATA
# ==========================================

print("Loading resume dataset...")

resume = pd.read_csv("Data/raw/resume.csv")

print("Resume shape:", resume.shape)


# Remove unnecessary HTML column
resume = resume.drop(columns=["Resume_html"])


# Remove duplicate rows
resume = resume.drop_duplicates()


# Remove rows where resume text or category is empty
resume = resume.dropna(subset=["Resume_str", "Category"])


# Clean resume text
def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


resume["Resume_str"] = resume["Resume_str"].apply(clean_text)


# Save processed resume dataset
resume.to_csv(
    "Data/processed/resume_clean.csv",
    index=False
)

print("Clean resume shape:", resume.shape)
print("Saved: Data/processed/resume_clean.csv")


# ==========================================
# 2. LOAD JOB DATA
# ==========================================

print("\nLoading job dataset...")

jobs = pd.read_csv("Data/raw/indian_job.csv")

print("Job shape:", jobs.shape)


# Remove duplicate jobs
jobs = jobs.drop_duplicates(subset=["job_id"])


# Remove jobs without title
jobs = jobs.dropna(subset=["title"])


# Fill missing text fields
jobs["company_name"] = jobs["company_name"].fillna("Unknown Company")

jobs["description"] = jobs["description"].fillna("")

jobs["skills_desc"] = jobs["skills_desc"].fillna("")

jobs["location"] = jobs["location"].fillna("Unknown")

jobs["formatted_experience_level"] = (
    jobs["formatted_experience_level"].fillna("Not Specified")
)


# ==========================================
# CREATE COMBINED JOB TEXT
# ==========================================

jobs["job_text"] = (
    jobs["title"].fillna("") + " " +
    jobs["description"].fillna("") + " " +
    jobs["skills_desc"].fillna("") + " " +
    jobs["formatted_experience_level"].fillna("")
)


# Clean job text
jobs["job_text"] = jobs["job_text"].apply(clean_text)


# ==========================================
# SELECT IMPORTANT COLUMNS
# ==========================================

jobs_clean = jobs[
    [
        "job_id",
        "company_name",
        "title",
        "location",
        "formatted_experience_level",
        "job_text"
    ]
].copy()


# ==========================================
# SAVE PROCESSED JOB DATA
# ==========================================

jobs_clean.to_csv(
    "Data/processed/jobs_clean.csv",
    index=False
)

print("Clean job shape:", jobs_clean.shape)

print("Saved: Data/processed/jobs_clean.csv")


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n====================================")
print("DATA PREPROCESSING COMPLETED")
print("====================================")

print("\nProcessed files:")
print("1. Data/processed/resume_clean.csv")
print("2. Data/processed/jobs_clean.csv")