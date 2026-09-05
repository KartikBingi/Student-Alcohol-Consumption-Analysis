# ==============================================================================
# STREAMLIT DASHBOARD: STUDENT PERFORMANCE EDA
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Set Streamlit page layout and title
st.set_page_config(
    page_title="Student Performance EDA Dashboard",
    page_icon="📊",
    layout="wide",
)

# Set visual style for Seaborn plots
sns.set_theme(style="whitegrid")

# App Header
st.title("📊 Student Performance Exploratory Data Analysis")
st.markdown(
    "Interactive dashboard exploring social, demographic, and academic factors influencing student outcomes."
)


# ==============================================================================
# 1. DATA LOADING FUNCTION
# ==============================================================================
@st.cache_data
def load_data(mat_file, por_file):
    """Load and preprocess Mathematics and Portuguese datasets."""
    # Attempt parsing with comma first, fallback to semicolon
    try:
        mat_df = pd.read_csv(mat_file, sep=",")
        por_df = pd.read_csv(por_file, sep=",")
    except Exception:
        mat_df = pd.read_csv(mat_file, sep=";")
        por_df = pd.read_csv(por_file, sep=";")

    # Encode binary variables into numerical format (1 / 0)
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
        if col in mat_df.columns:
            mat_df[col + "_encoded"] = mat_df[col].map({"yes": 1, "no": 0})
        if col in por_df.columns:
            por_df[col + "_encoded"] = por_df[col].map({"yes": 1, "no": 0})

    return mat_df, por_df


# Sidebar File Uploaders
st.sidebar.header("1. Upload Datasets")
uploaded_mat = st.sidebar.file_uploader(
    "Upload student-mat.csv", type=["csv"]
)
uploaded_por = st.sidebar.file_uploader(
    "Upload student-por.csv", type=["csv"]
)

# Load default local files if no user upload provided
if uploaded_mat is not None and uploaded_por is not None:
    mat_df, por_df = load_data(uploaded_mat, uploaded_por)
else:
    try:
        mat_df, por_df = load_data("student-mat.csv", "student-por.csv")
        st.sidebar.info("Using default local datasets.")
    except Exception:
        st.error(
            "Please upload both 'student-mat.csv' and 'student-por.csv' files in the sidebar to proceed."
        )
        st.stop()


# ==============================================================================
# 2. SIDEBAR FILTERS & DATASET SELECTION
# ==============================================================================
st.sidebar.header("2. Dashboard Filters")

# Select active dataset for detailed view
dataset_choice = st.sidebar.radio(
    "Select Subject Dataset:", ("Mathematics", "Portuguese")
)
active_df = mat_df if dataset_choice == "Mathematics" else por_df

# Interactive sidebar filters
selected_school = st.sidebar.multiselect(
    "Filter by School:",
    options=active_df["school"].unique(),
    default=active_df["school"].unique(),
)

selected_sex = st.sidebar.multiselect(
    "Filter by Gender:",
    options=active_df["sex"].unique(),
    default=active_df["sex"].unique(),
)

# Apply active filters
filtered_df = active_df[
    (active_df["school"].isin(selected_school))
    & (active_df["sex"].isin(selected_sex))
]


# ==============================================================================
# 3. KEY PERFORMANCE INDICATORS (KPIs)
# ==============================================================================
st.subheader(f"📌 Overview Metrics ({dataset_choice})")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(filtered_df))
col2.metric("Average Final Grade (G3)", f"{filtered_df['G3'].mean():.2f} / 20")
col3.metric("Pass Rate (G3 >= 10)", f"{(filtered_df['G3'] >= 10).mean()*100:.1f}%")
col4.metric("Avg Absences", f"{filtered_df['absences'].mean():.1f} days")

st.markdown("---")


# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Grade Distributions",
        "🔥 Correlation Heatmap",
        "🎯 Feature Relationships",
        "📋 Data Table & Stats",
    ]
)

# Tab 1: Grade Distributions
with tab1:
    st.subheader("Distribution of Final Grades (G3)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        filtered_df["G3"],
        kde=True,
        bins=20,
        color="skyblue" if dataset_choice == "Mathematics" else "orange",
        ax=ax,
    )
    ax.set_title(f"Final Grade Distribution - {dataset_choice}")
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Student Count")
    st.pyplot(fig)

# Tab 2: Correlation Heatmap
with tab2:
    st.subheader("Pairwise Feature Correlation")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation Heatmap - {dataset_choice}")
    st.pyplot(fig)

# Tab 3: Categorical Feature Impact
with tab3:
    st.subheader("Key Academic Drivers")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Past Failures vs Final Grade**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="failures", y="G3", data=filtered_df, palette="Set2", ax=ax1
        )
        ax1.set_xlabel("Number of Past Failures")
        ax1.set_ylabel("Final Grade (G3)")
        st.pyplot(fig1)

    with col_b:
        st.markdown("**Higher Education Ambition vs Final Grade**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="higher", y="G3", data=filtered_df, palette="Pastel1", ax=ax2
        )
        ax2.set_xlabel("Desire for Higher Education")
        ax2.set_ylabel("Final Grade (G3)")
        st.pyplot(fig2)

# Tab 4: Raw Data & Descriptive Statistics
with tab4:
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered_df.describe().T)

    st.subheader("Filtered Raw Data View")
    st.dataframe(filtered_df)