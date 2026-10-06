import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from resume_parser import extract_resume_text
from recommender import recommend_jobs
from skill_gap import analyze_skill_gap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background: #f7f9fc;
    }


    /* Hide Streamlit default menu */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Main title */

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }


    .subtitle {
        font-size: 20px;
        color: #667085;
        margin-bottom: 25px;
    }


    /* Cards */

    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e4e7ec;
        box-shadow: 0 5px 20px rgba(16, 24, 40, 0.06);
        margin-bottom: 20px;
    }


    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e4e7ec;
        height: 190px;
        box-shadow: 0 5px 20px rgba(16, 24, 40, 0.05);
    }


    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 24, 40, 0.05);
    }


    .metric-value {
        font-size: 32px;
        font-weight: 800;
    }


    .metric-label {
        color: #667085;
        font-size: 14px;
    }


    .job-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        margin-bottom: 15px;
    }


    .skill-match {
        display: inline-block;
        background: #ecfdf3;
        color: #027a48;
        padding: 7px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
        font-weight: 600;
    }


    .skill-gap {
        display: inline-block;
        background: #fff1f3;
        color: #c01048;
        padding: 7px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
        font-weight: 600;
    }


    .step-box {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        text-align: center;
    }


    .small-text {
        color: #667085;
        font-size: 14px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "username" not in st.session_state:
    st.session_state.username = ""

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""

if "category" not in st.session_state:
    st.session_state.category = ""

if "top_jobs" not in st.session_state:
    st.session_state.top_jobs = None

if "skill_gap_result" not in st.session_state:
    st.session_state.skill_gap_result = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# ============================================================
# LOAD CLASSIFIER
# ============================================================

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "classifier.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)


@st.cache_resource
def load_classifier():

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return classifier, vectorizer


# ============================================================
# PREDICT RESUME CATEGORY
# ============================================================

def predict_category(resume_text):

    classifier, vectorizer = (
        load_classifier()
    )

    resume_vector = (
        vectorizer.transform(
            [resume_text]
        )
    )

    prediction = classifier.predict(
        resume_vector
    )

    return prediction[0]


# ============================================================
# FAST JOB RECOMMENDATION CACHE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def cached_recommend_jobs(
    resume_text,
    top_n=10
):

    return recommend_jobs(
        resume_text,
        top_n=top_n
    )


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.logged_in = False
    st.session_state.page = "login"
    st.session_state.username = ""

    st.session_state.resume_text = ""
    st.session_state.resume_name = ""
    st.session_state.category = ""
    st.session_state.top_jobs = None
    st.session_state.skill_gap_result = None
    st.session_state.analysis_complete = False

    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <h1>💼 SmartHire</h1>
            <p class="small-text">
            Resume-to-Job Matching & Career Guidance
            </p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        if st.session_state.logged_in:

            st.write(
                f"👤 Welcome, **{st.session_state.username}**"
            )

            st.divider()

            if st.button(
                "🏠 Home",
                use_container_width=True
            ):
                st.session_state.page = "intro"
                st.rerun()


            if st.button(
                "📄 Upload Resume",
                use_container_width=True
            ):
                st.session_state.page = "upload"
                st.rerun()


            if st.button(
                "📊 Dashboard",
                use_container_width=True
            ):

                if st.session_state.analysis_complete:
                    st.session_state.page = "dashboard"
                    st.rerun()

                else:
                    st.warning(
                        "Analyze a resume first."
                    )


            st.divider()


            if st.button(
                "🚪 Logout",
                use_container_width=True
            ):
                logout()


# ============================================================
# PAGE 1 — LOGIN
# ============================================================

def login_page():

    st.markdown(
        "<div style='height:70px'></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1, 1.5, 1]
    )

    with col2:

        st.markdown(
            """
            <div style="
                text-align:center;
                background:white;
                padding:40px;
                border-radius:22px;
                border:1px solid #e4e7ec;
                box-shadow:0 10px 30px rgba(16,24,40,.08);
            ">

            <div style="font-size:60px;">💼</div>

            <h1>SmartHire</h1>

            <p style="color:#667085;">
            Resume-to-Job Matching & Career Guidance
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        username = st.text_input(
            "👤 Username"
        )

        password = st.text_input(
            "🔐 Password",
            type="password"
        )

        st.info(
            "Demo login: username = admin | password = smarthire"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True
        ):

            if (
                username == "admin"
                and password == "smarthire"
            ):

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "intro"

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )


# ============================================================
# PAGE 2 — INTRODUCTION
# ============================================================

def intro_page():

    st.markdown(
        "<div class='main-title'>Welcome to SmartHire 🚀</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Your Resume → Your Skills → Your Career Opportunities
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <h2>What is SmartHire?</h2>

        <p>
        SmartHire is a Machine Learning based career guidance
        system that analyzes a candidate's resume, predicts
        the resume category, recommends suitable jobs and
        identifies important skill gaps.
        </p>

        <p>
        The system uses classical Machine Learning techniques
        such as TF-IDF, Logistic Regression, Cosine Similarity
        and K-Means clustering.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "✨ What SmartHire Can Do"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <h3>📄</h3>
            <h3>Resume Analysis</h3>

            <p class="small-text">
            Extract information from PDF,
            DOCX or TXT resumes.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <h3>🤖</h3>
            <h3>ML Classification</h3>

            <p class="small-text">
            Predict the candidate's
            resume category.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="feature-card">

            <h3>🎯</h3>
            <h3>Job Matching</h3>

            <p class="small-text">
            Find jobs using
            TF-IDF similarity.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="feature-card">

            <h3>🧠</h3>
            <h3>Skill Gap</h3>

            <p class="small-text">
            Identify skills to improve
            for the target job.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")
    st.write("")


    st.subheader(
        "🔬 SmartHire Technology"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.markdown(
            "<div class='step-box'>🐍<br><b>Python</b></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            "<div class='step-box'>📊<br><b>Scikit-learn</b></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            "<div class='step-box'>🧮<br><b>TF-IDF</b></div>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            "<div class='step-box'>🌐<br><b>Streamlit</b></div>",
            unsafe_allow_html=True
        )


    st.write("")


    if st.button(
        "🚀 Get Started",
        use_container_width=True
    ):

        st.session_state.page = "upload"
        st.rerun()


# ============================================================
# PAGE 3 — RESUME UPLOAD
# ============================================================

def upload_page():

    st.markdown(
        "<div class='main-title'>📄 Upload Your Resume</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Upload your resume and let SmartHire analyze your career profile.
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX and TXT"
    )


    if uploaded_file is not None:

        st.success(
            f"Resume uploaded: {uploaded_file.name}"
        )

        file_size = (
            uploaded_file.size / 1024
        )

        st.write(
            f"📎 File size: {file_size:.2f} KB"
        )


        if st.button(
            "🚀 Analyze My Resume",
            use_container_width=True
        ):

            suffix = Path(
                uploaded_file.name
            ).suffix.lower()


            progress = st.progress(
                0
            )

            status = st.empty()


            temp_path = None


            try:

                # ------------------------------------------
                # SAVE TEMPORARY RESUME
                # ------------------------------------------

                status.info(
                    "📄 Preparing your resume..."
                )

                progress.progress(
                    15
                )


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = Path(
                        temp_file.name
                    )


                # ------------------------------------------
                # EXTRACT TEXT
                # ------------------------------------------

                status.info(
                    "📝 Extracting resume information..."
                )

                progress.progress(
                    30
                )


                resume_text = (
                    extract_resume_text(
                        temp_path
                    )
                )


                if not resume_text.strip():

                    raise ValueError(
                        "No readable text was found in the resume."
                    )


                # ------------------------------------------
                # CLASSIFICATION
                # ------------------------------------------

                status.info(
                    "🤖 Predicting your career category..."
                )

                progress.progress(
                    50
                )


                category = predict_category(
                    resume_text
                )


                # ------------------------------------------
                # JOB RECOMMENDATION
                # ------------------------------------------

                status.info(
                    "🎯 Finding matching jobs..."
                )

                progress.progress(
                    65
                )


                top_jobs = cached_recommend_jobs(
                    resume_text,
                    10
                )


                # ------------------------------------------
                # SKILL GAP
                # ------------------------------------------

                status.info(
                    "🧠 Analyzing your skill gap..."
                )

                progress.progress(
                    82
                )


                skill_gap_result = (
                    analyze_skill_gap(
                        resume_text,
                        top_jobs
                    )
                )


                # ------------------------------------------
                # SAVE SESSION DATA
                # ------------------------------------------

                st.session_state.resume_text = (
                    resume_text
                )

                st.session_state.resume_name = (
                    uploaded_file.name
                )

                st.session_state.category = (
                    category
                )

                st.session_state.top_jobs = (
                    top_jobs
                )

                st.session_state.skill_gap_result = (
                    skill_gap_result
                )

                st.session_state.analysis_complete = (
                    True
                )


                progress.progress(
                    100
                )

                status.success(
                    "✅ Analysis completed successfully!"
                )


                # ------------------------------------------
                # CLEAN TEMP FILE
                # ------------------------------------------

                if (
                    temp_path
                    and temp_path.exists()
                ):

                    temp_path.unlink()


                st.session_state.page = (
                    "dashboard"
                )

                st.rerun()


            except Exception as error:

                if (
                    temp_path
                    and temp_path.exists()
                ):

                    try:
                        temp_path.unlink()
                    except Exception:
                        pass


                st.error(
                    f"Something went wrong: {error}"
                )


    st.write("")


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "← Back to Introduction",
            use_container_width=True
        ):

            st.session_state.page = "intro"
            st.rerun()


    with col2:

        if (
            st.session_state.analysis_complete
        ):

            if st.button(
                "View Previous Analysis →",
                use_container_width=True
            ):

                st.session_state.page = (
                    "dashboard"
                )

                st.rerun()


# ============================================================
# PAGE 4 — CAREER DASHBOARD
# ============================================================

def dashboard_page():

    if not st.session_state.analysis_complete:

        st.warning(
            "Please upload and analyze a resume first."
        )

        if st.button(
            "📄 Upload Resume"
        ):

            st.session_state.page = "upload"
            st.rerun()

        return


    category = (
        st.session_state.category
    )

    top_jobs = (
        st.session_state.top_jobs
    )

    skill_gap = (
        st.session_state.skill_gap_result
    )


    st.markdown(
        "<div class='main-title'>📊 Career Dashboard</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="subtitle">
        Analysis for <b>{st.session_state.resume_name}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    top_match = 0

    if (
        top_jobs is not None
        and len(top_jobs) > 0
    ):

        top_match = float(
            top_jobs.iloc[0]["match_score"]
        )


    skill_match = 0

    if skill_gap:

        skill_match = float(
            skill_gap.get(
                "skill_match_percentage",
                0
            )
        )


    missing_count = 0

    if skill_gap:

        missing_count = len(
            skill_gap.get(
                "missing_skills",
                []
            )
        )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-value">
            🤖
            </div>

            <div>
            <b>{category}</b>
            </div>

            <div class="metric-label">
            Resume Category
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-value">
            {top_match:.1f}%
            </div>

            <div class="metric-label">
            Top Job Match
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-value">
            {skill_match:.1f}%
            </div>

            <div class="metric-label">
            Skill Match
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-value">
            {missing_count}
            </div>

            <div class="metric-label">
            Skills to Improve
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # JOB RECOMMENDATIONS
    # ========================================================

    st.markdown(
        "<div class='card'><h2>🎯 Recommended Jobs</h2></div>",
        unsafe_allow_html=True
    )


    if top_jobs is not None:

        for rank, (_, job) in enumerate(
            top_jobs.iterrows(),
            start=1
        ):

            score = float(
                job.get(
                    "match_score",
                    0
                )
            )


            st.markdown(
                f"""
                <div class="job-card">

                <h3>
                {rank}. {job.get("title", "Unknown Job")}
                </h3>

                <p>
                🏢 <b>
                {job.get("company_name", "Unknown Company")}
                </b>
                </p>

                <p>
                📍 {job.get("location", "Not specified")}
                &nbsp;&nbsp;
                💼 {job.get("formatted_experience_level", "Not specified")}
                </p>

                <p>
                🎯 <b>Match Score: {score:.2f}%</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SKILL ANALYSIS
    # ========================================================

    st.markdown(
        "<div class='card'><h2>🧠 Skill Analysis</h2></div>",
        unsafe_allow_html=True
    )


    if skill_gap:

        matched = skill_gap.get(
            "matched_skills",
            []
        )

        missing = skill_gap.get(
            "missing_skills",
            []
        )


        st.subheader(
            "✅ Your Matching Skills"
        )


        if matched:

            matched_html = ""

            for skill in matched:

                matched_html += (
                    f'<span class="skill-match">'
                    f'✓ {skill}'
                    f'</span>'
                )

            st.markdown(
                matched_html,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No matching skills detected."
            )


        st.write("")


        st.subheader(
            "📚 Skills to Improve"
        )


        if missing:

            missing_html = ""

            for skill in missing:

                missing_html += (
                    f'<span class="skill-gap">'
                    f'→ {skill}'
                    f'</span>'
                )

            st.markdown(
                missing_html,
                unsafe_allow_html=True
            )

        else:

            st.success(
                "No major skill gaps found."
            )


        st.write("")


        st.subheader(
            "Skill Match Progress"
        )

        st.progress(
            min(
                max(
                    skill_match / 100,
                    0
                ),
                1
            )
        )

        st.write(
            f"Your skill match is **{skill_match:.2f}%**"
        )


    # ========================================================
    # JOB CLUSTERING
    # ========================================================

    st.markdown(
        "<div class='card'><h2>🔬 Job Market Clusters</h2></div>",
        unsafe_allow_html=True
    )


    cluster_file = (
        BASE_DIR
        / "Data"
        / "processed"
        / "jobs_clustered.csv"
    )


    if cluster_file.exists():

        try:

            clustered_jobs = pd.read_csv(
                cluster_file
            )


            if "cluster" in clustered_jobs.columns:

                cluster_counts = (
                    clustered_jobs[
                        "cluster"
                    ]
                    .value_counts()
                    .sort_index()
                )


                st.write(
                    "SmartHire grouped job postings "
                    "into clusters using K-Means."
                )


                cluster_df = pd.DataFrame(
                    {
                        "Cluster": cluster_counts.index,
                        "Number of Jobs": cluster_counts.values
                    }
                )


                st.dataframe(
                    cluster_df,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as error:

            st.info(
                f"Cluster information unavailable: {error}"
            )

    else:

        st.info(
            "Job clustering data is not available yet."
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.write("")

    st.markdown(
        "<div class='card'><h2>📥 Career Report</h2></div>",
        unsafe_allow_html=True
    )


    report_data = {
        "Resume": [
            st.session_state.resume_name
        ],
        "Resume Category": [
            category
        ],
        "Top Job": [
            skill_gap.get(
                "job_title",
                "Not available"
            )
            if skill_gap
            else "Not available"
        ],
        "Top Match Score": [
            top_match
        ],
        "Skill Match Percentage": [
            skill_match
        ],
        "Matched Skills": [
            ", ".join(
                skill_gap.get(
                    "matched_skills",
                    []
                )
            )
            if skill_gap
            else ""
        ],
        "Missing Skills": [
            ", ".join(
                skill_gap.get(
                    "missing_skills",
                    []
                )
            )
            if skill_gap
            else ""
        ]
    }


    report_df = pd.DataFrame(
        report_data
    )


    csv_data = report_df.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download Career Report",
        data=csv_data,
        file_name="SmartHire_Career_Report.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # NAVIGATION
    # ========================================================

    st.write("")

    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "📄 Analyze Another Resume",
            use_container_width=True
        ):

            st.session_state.page = "upload"
            st.rerun()


    with col2:

        if st.button(
            "🏠 Back to Home",
            use_container_width=True
        ):

            st.session_state.page = "intro"
            st.rerun()


# ============================================================
# SIDEBAR + PAGE ROUTER
# ============================================================

show_sidebar()


if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.page == "intro":

        intro_page()

    elif st.session_state.page == "upload":

        upload_page()

    elif st.session_state.page == "dashboard":

        dashboard_page()

    else:

        intro_page()