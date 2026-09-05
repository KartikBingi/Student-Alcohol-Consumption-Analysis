# ==============================================================================
# LIVE INTERACTIVE STREAMLIT DASHBOARD: STUDENT PERFORMANCE EDA
# ==============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Mode High-Contrast Cards & UI Elements
st.markdown(
    """
    <style>
    .main { padding: 1rem; }
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads, combines, and preprocesses Math and Portuguese datasets."""
    try:
        mat = pd.read_csv("student-mat.csv", sep=",")
        por = pd.read_csv("student-por.csv", sep=",")
    except Exception:
        mat = pd.read_csv("student-mat.csv", sep=";")
        por = pd.read_csv("student-por.csv", sep=";")

    # Add course subject flags
    mat["Subject"] = "Mathematics"
    por["Subject"] = "Portuguese"

    # Concatenate into unified dataset
    combined_df = pd.concat([mat, por], ignore_index=True)

    # Encode binary variables into numeric flags
    binary_cols = [
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    ]
    for col in binary_cols:
        if col in combined_df.columns:
            combined_df[col + "_encoded"] = combined_df[col].map(
                {"yes": 1, "no": 0}
            )

    return combined_df


df = load_data()


# ------------------------------------------------------------------------------
# 3. SIDEBAR INTERACTIVE FILTERS
# ------------------------------------------------------------------------------
st.sidebar.title("🔍 Interactive Filters")

# Subject Course Selection
subject_choice = st.sidebar.radio(
    "Select Subject:",
    options=["All Subjects", "Mathematics", "Portuguese"],
    index=0,
)

if subject_choice != "All Subjects":
    filtered_df = df[df["Subject"] == subject_choice]
else:
    filtered_df = df.copy()

# Filter by School
selected_school = st.sidebar.multiselect(
    "School:",
    options=filtered_df["school"].unique(),
    default=filtered_df["school"].unique(),
)

# Filter by Gender
selected_sex = st.sidebar.multiselect(
    "Gender:",
    options=filtered_df["sex"].unique(),
    default=filtered_df["sex"].unique(),
)

# Filter by Age Range
age_range = st.sidebar.slider(
    "Age Range:",
    min_value=int(filtered_df["age"].min()),
    max_value=int(filtered_df["age"].max()),
    value=(int(filtered_df["age"].min()), int(filtered_df["age"].max())),
)

# Apply Active Sidebar Filters
filtered_df = filtered_df[
    (filtered_df["school"].isin(selected_school))
    & (filtered_df["sex"].isin(selected_sex))
    & (filtered_df["age"].between(age_range[0], age_range[1]))
]


# ------------------------------------------------------------------------------
# 4. DASHBOARD HEADER & KPI METRIC CARDS
# ------------------------------------------------------------------------------
st.title("📊 Student Performance Interactive Analytics")
st.markdown(
    "Explore factors influencing final grades (`G3`) across secondary school courses."
)

# Render 4 Overview Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students Selected", f"{len(filtered_df)}")
col2.metric("Mean Final Grade (G3)", f"{filtered_df['G3'].mean():.2f} / 20")
col3.metric("Pass Rate (G3 ≥ 10)", f"{(filtered_df['G3'] >= 10).mean() * 100:.1f}%")
col4.metric("Avg School Absences", f"{filtered_df['absences'].mean():.1f} Days")

st.markdown("---")


# ------------------------------------------------------------------------------
# 5. VISUAL TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Grade Distributions",
        "🔥 Interactive Heatmap",
        "🎯 Academic Indicators",
        "📋 Data Explorer",
    ]
)

# TAB 1: Distribution Analysis
with tab1:
    st.subheader("Final Grade (G3) Distribution")
    fig_hist = px.histogram(
        filtered_df,
        x="G3",
        color="Subject" if subject_choice == "All Subjects" else "sex",
        marginal="box",
        nbins=20,
        title="Distribution & Boxplot of Final Grades (G3)",
        labels={"G3": "Final Grade (0-20)"},
        template="plotly_dark",
        barmode="overlay",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2: Heatmap Correlation Analysis
with tab2:
    st.subheader("Feature Correlation Matrix")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix Heatmap",
        template="plotly_dark",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 3: Academic & Social Impact Drivers
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Past Class Failures vs Final Grade")
        fig_failures = px.box(
            filtered_df,
            x="failures",
            y="G3",
            color="failures",
            title="Impact of Past Failures on G3 Score",
            labels={
                "failures": "Number of Past Failures",
                "G3": "Final Grade",
            },
            template="plotly_dark",
        )
        st.plotly_chart(fig_failures, use_container_width=True)

    with col_b:
        st.subheader("Higher Education Ambition vs Final Grade")
        fig_higher = px.violin(
            filtered_df,
            x="higher",
            y="G3",
            color="higher",
            box=True,
            points="all",
            title="Higher Education Aspirations vs G3 Score",
            labels={"higher": "Wants Higher Education", "G3": "Final Grade"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_higher, use_container_width=True)

# TAB 4: Raw Data & Export Option
with tab4:
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe().T, use_container_width=True)

    st.subheader("Filtered Dataset View")
    st.dataframe(filtered_df, use_container_width=True)

    # Download CSV feature
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_student_data.csv",
        mime="text/csv",
    )