# ==============================================================================
# LIVE INTERACTIVE STREAMLIT DASHBOARD: STUDENT PERFORMANCE EDA
# ==============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg

# Configure Streamlit page settings
st.set_page_config(
    page_title="Student Performance EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================
@st.cache_data
def load_data():
    """Loads and preprocesses student performance datasets."""
    # Read datasets with automatic delimiter fallback
    try:
        mat = pd.read_csv("student-mat.csv", sep=",")
        por = pd.read_csv("student-por.csv", sep=",")
    except Exception:
        mat = pd.read_csv("student-mat.csv", sep=";")
        por = pd.read_csv("student-por.csv", sep=";")

    # Add subject indicator column
    mat["Subject"] = "Mathematics"
    por["Subject"] = "Portuguese"

    # Combine datasets for unified exploration
    combined_df = pd.concat([mat, por], ignore_index=True)

    # Encode binary variables
    binary_cols = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
    for col in binary_cols:
        if col in combined_df.columns:
            combined_df[col + "_encoded"] = combined_df[col].map({"yes": 1, "no": 0})

    return combined_df

df = load_data()


# ==============================================================================
# 2. SIDEBAR CONTROLS & INTERACTIVE FILTERS
# ==============================================================================
st.sidebar.title("🔍 Interactive Filters")

# Subject Filter
subject_choice = st.sidebar.radio(
    "Select Subject:",
    options=["All Subjects", "Mathematics", "Portuguese"],
    index=0
)

# Filter Dataset based on Subject choice
if subject_choice != "All Subjects":
    filtered_df = df[df["Subject"] == subject_choice]
else:
    filtered_df = df.copy()

# Dynamic Multiselect Filters
selected_school = st.sidebar.multiselect(
    "School:",
    options=filtered_df["school"].unique(),
    default=filtered_df["school"].unique()
)

selected_sex = st.sidebar.multiselect(
    "Gender:",
    options=filtered_df["sex"].unique(),
    default=filtered_df["sex"].unique()
)

age_range = st.sidebar.slider(
    "Age Range:",
    min_value=int(filtered_df["age"].min()),
    max_value=int(filtered_df["age"].max()),
    value=(int(filtered_df["age"].min()), int(filtered_df["age"].max()))
)

# Apply active filters
filtered_df = filtered_df[
    (filtered_df["school"].isin(selected_school)) &
    (filtered_df["sex"].isin(selected_sex)) &
    (filtered_df["age"].between(age_range[0], age_range[1]))
]


# ==============================================================================
# 3. DASHBOARD HEADER & KPI METRICS
# ==============================================================================
st.title("📊 Student Performance Interactive Analytics")
st.markdown("Explore factors influencing final grades (`G3`) across secondary school courses.")

# Display Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students Selected", f"{len(filtered_df)}")
col2.metric("Mean Final Grade (G3)", f"{filtered_df['G3'].mean():.2f} / 20")
col3.metric("Pass Rate (G3 ≥ 10)", f"{(filtered_df['G3'] >= 10).mean() * 100:.1f}%")
col4.metric("Avg School Absences", f"{filtered_df['absences'].mean():.1f} Days")

st.markdown("---")


# ==============================================================================
# 4. INTERACTIVE VISUAL TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Grade Distributions", 
    "🔥 Interactive Heatmap", 
    "🎯 Academic Indicators", 
    "📋 Data Explorer"
])

# TAB 1: Live Interactive Grade Distribution
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
        template="plotly_white",
        barmode="overlay"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2: Interactive Correlation Matrix
with tab2:
    st.subheader("Feature Correlation Matrix")
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 3: Academic Drivers
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
            labels={"failures": "Number of Past Failures", "G3": "Final Grade"},
            template="plotly_white"
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
            template="plotly_white"
        )
        st.plotly_chart(fig_higher, use_container_width=True)

# TAB 4: Data Table & Download
with tab4:
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe().T, use_container_width=True)

    st.subheader("Filtered Dataset View")
    st.dataframe(filtered_df, use_container_width=True)

    # Live CSV Export
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )