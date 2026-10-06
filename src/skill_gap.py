import pandas as pd
import re


# --------------------------------------------------
# COMMON SKILL LIST
# --------------------------------------------------

skill_list = [
    "python",
    "java",
    "javascript",
    "c++",
    "c#",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data analysis",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "matplotlib",
    "seaborn",
    "power bi",
    "tableau",
    "excel",
    "statistics",
    "nlp",
    "natural language processing",
    "computer vision",
    "cloud computing",
    "aws",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "fastapi",
    "flask",
    "django",
    "spark",
    "hadoop",
    "data visualization",
    "business intelligence",
    "data engineering",
    "etl",
    "communication",
    "project management",
    "leadership"
]


# --------------------------------------------------
# FIND SKILLS FROM TEXT
# --------------------------------------------------

def find_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill in skill_list:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# --------------------------------------------------
# SKILL GAP ANALYSIS
# --------------------------------------------------

def analyze_skill_gap(resume_text, jobs):

    if not resume_text or not resume_text.strip():

        raise ValueError("Resume text is empty.")

    if jobs.empty:

        raise ValueError("No recommended jobs available.")

    # --------------------------------------------------
    # 1. FIND USER SKILLS FROM RESUME
    # --------------------------------------------------

    user_skills = find_skills(resume_text)

    user_skill_set = set(user_skills)

    # --------------------------------------------------
    # 2. GET TOP RECOMMENDED JOB
    # --------------------------------------------------

    top_job = jobs.iloc[0]

    job_title = top_job.get(
        "title",
        "Unknown Job"
    )

    company = top_job.get(
        "company_name",
        "Unknown Company"
    )

    match_score = top_job.get(
        "match_score",
        0
    )

    job_text = top_job.get(
        "job_text",
        ""
    )

    # --------------------------------------------------
    # 3. FIND REQUIRED SKILLS
    # --------------------------------------------------

    required_skills = find_skills(job_text)

    required_skill_set = set(required_skills)

    # --------------------------------------------------
    # 4. COMPARE SKILLS
    # --------------------------------------------------

    matched_skills = (
        user_skill_set.intersection(
            required_skill_set
        )
    )

    missing_skills = (
        required_skill_set.difference(
            user_skill_set
        )
    )

    # --------------------------------------------------
    # 5. CALCULATE SKILL MATCH %
    # --------------------------------------------------

    if len(required_skill_set) > 0:

        skill_match_percentage = (
            len(matched_skills)
            / len(required_skill_set)
        ) * 100

    else:

        skill_match_percentage = 0

    # --------------------------------------------------
    # 6. DISPLAY RESULT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SKILL GAP ANALYSIS")
    print("=" * 70)

    print("\nSkills detected from your resume:")

    if user_skills:

        for skill in user_skills:

            print("-", skill)

    else:

        print("No predefined skills detected.")

    print("\n" + "-" * 60)

    print("\nTARGET JOB")

    print("Job Title:", job_title)
    print("Company:", company)
    print(f"Job Match Score: {match_score:.2f}%")

    print("\nRequired Skills:")

    if required_skills:

        for skill in required_skills:

            print("-", skill)

    else:

        print("No predefined skills found.")

    print("\nMatched Skills:")

    if matched_skills:

        for skill in sorted(matched_skills):

            print("✓", skill)

    else:

        print("No matching skills found.")

    print("\nSkills to Improve / Skill Gap:")

    if missing_skills:

        for skill in sorted(missing_skills):

            print("→", skill)

    else:

        print("No major skill gaps found.")

    print("\nSkill Match Percentage:")

    print(
        f"{skill_match_percentage:.2f}%"
    )

    # --------------------------------------------------
    # 7. SAVE REPORT
    # --------------------------------------------------

    report_data = {

        "job_title": [job_title],

        "company": [company],

        "match_score": [match_score],

        "skill_match_percentage": [
            skill_match_percentage
        ],

        "matched_skills": [
            ", ".join(
                sorted(matched_skills)
            )
        ],

        "missing_skills": [
            ", ".join(
                sorted(missing_skills)
            )
        ]
    }

    report = pd.DataFrame(
        report_data
    )

    output_path = (
        "Data/processed/"
        "skill_gap_report.csv"
    )

    report.to_csv(
        output_path,
        index=False
    )

    print("\nReport saved to:")

    print(output_path)

    print("\n" + "=" * 70)
    print("SKILL GAP ANALYSIS COMPLETED")
    print("=" * 70)

    return {

        "user_skills": user_skills,

        "required_skills": required_skills,

        "matched_skills": sorted(
            matched_skills
        ),

        "missing_skills": sorted(
            missing_skills
        ),

        "skill_match_percentage":
            skill_match_percentage,

        "job_title": job_title,

        "company": company,

        "match_score": match_score
    }