import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv


# ----------------------------------------------------------------------------
# Setup: database connection
# ----------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL is missing from the .env file.")
    st.stop()

engine = create_engine(DATABASE_URL)
def run_query(query):
    connection = engine.raw_connection()
    try:
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Styling (CSS injected via st.markdown)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall page */
        .main {
            background-color: #f7f9fb;
        }

        /* Header */
        .dashboard-header {
            padding: 1.25rem 1.5rem;
            background-color: #ffffff;
            border: 1px solid #e3e7ed;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        .dashboard-title {
            font-size: 1.9rem;
            font-weight: 700;
            color: #10233f;
            margin-bottom: 0.15rem;
        }
        .dashboard-subtitle {
            font-size: 0.95rem;
            color: #5b6b82;
            margin-bottom: 0.5rem;
        }
        .part-indicator {
            display: inline-block;
            background-color: #eaf1ff;
            color: #1a4fa3;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
        }

        /* Section headings */
        .section-heading {
            font-size: 1.25rem;
            font-weight: 700;
            color: #10233f;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #1a4fa3;
            padding-left: 0.6rem;
        }
        .section-subheading {
            font-size: 1.02rem;
            font-weight: 600;
            color: #26364b;
            margin-top: 0.75rem;
            margin-bottom: 0.4rem;
        }

        /* KPI cards */
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e3e7ed;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            text-align: left;
        }
        .kpi-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #5b6b82;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #10233f;
        }

        /* Chart card wrapper */
        .chart-card {
            background-color: #ffffff;
            border: 1px solid #e3e7ed;
            border-radius: 12px;
            padding: 0.75rem 1rem 0.25rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }

        /* Footer */
        .dashboard-footer {
            text-align: center;
            color: #8896a8;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e3e7ed;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
        <span class="part-indicator">Part 1 · Data Engineering &amp; Analytics</span>
        <div class="dashboard-title">Hospital Readmission Analytics</div>
        <div class="dashboard-subtitle">
            Interactive analysis of hospital admissions and 30-day readmission patterns
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this dashboard")
    st.write(
        "This dashboard summarizes hospital admission and 30-day readmission "
        "data from the analytical PostgreSQL database. All figures are "
        "computed directly from the underlying tables."
    )
    st.markdown("---")
    st.markdown("**Sections**")
    st.markdown(
        "- Overview\n"
        "- Readmission Insights\n"
        "- Hospital Utilization"
    )
    st.markdown("---")
    st.caption("Hospital Readmission Analytics — Academic Project — Part 1")


# ----------------------------------------------------------------------------
# Reusable chart styling helper
# ----------------------------------------------------------------------------
def style_fig(fig, show_legend=False):
    """Apply a consistent, clean visual theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        showlegend=show_legend,
        margin=dict(l=10, r=10, t=45, b=10),
        height=380,
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#26364b"),
        title_font=dict(size=15, color="#10233f"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5")
    return fig


# ----------------------------------------------------------------------------
# Overview: KPI section
# ----------------------------------------------------------------------------
st.markdown('<div class="section-heading">Overview</div>', unsafe_allow_html=True)

try:
    kpi_query = """
        SELECT
            COUNT(*) AS total_encounters,
            ROUND(AVG(length_of_stay_days), 2) AS avg_length_of_stay
        FROM admissions;
    """

    readmission_query = """
        SELECT
            ROUND(
                100.0 * SUM(readmitted_within_30_days) / COUNT(*),
                2
            ) AS readmission_rate
        FROM readmissions;
    """

    kpi_df = run_query(kpi_query)
    readmission_df = run_query(readmission_query)

    total_encounters = int(kpi_df.loc[0, "total_encounters"])
    avg_length_of_stay = float(kpi_df.loc[0, "avg_length_of_stay"])
    readmission_rate = float(readmission_df.loc[0, "readmission_rate"])

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Encounters</div>
                <div class="kpi-value">{total_encounters:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">30-Day Readmission Rate</div>
                <div class="kpi-value">{readmission_rate:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Length of Stay</div>
                <div class="kpi-value">{avg_length_of_stay:.2f} days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()


# ----------------------------------------------------------------------------
# Readmission Insights
# ----------------------------------------------------------------------------
st.markdown('<div class="section-heading">Readmission Insights</div>', unsafe_allow_html=True)

insight_col1, insight_col2 = st.columns(2)

# --- Readmission rate by age group ------------------------------------------------
age_query = """
    SELECT
        age_group,
        ROUND(
            100.0 * SUM(readmitted_within_30_days) / COUNT(*),
            2
        ) AS readmission_rate
    FROM (
        SELECT
            CASE
                WHEN age IN ('0-10', '10-20', '20-30') THEN '<30'
                WHEN age IN ('30-40', '40-50') THEN '30-49'
                WHEN age IN ('50-60', '60-70') THEN '50-69'
                WHEN age IN ('70-80', '80-90', '90-100') THEN '70+'
                ELSE 'Unknown'
            END AS age_group,
            r.readmitted_within_30_days
        FROM admissions a
        JOIN readmissions r
            ON a.encounter_id = r.encounter_id
    ) AS grouped_data
    GROUP BY age_group
    ORDER BY
        CASE age_group
            WHEN '<30' THEN 1
            WHEN '30-49' THEN 2
            WHEN '50-69' THEN 3
            WHEN '70+' THEN 4
            ELSE 5
        END;
"""

age_df = run_query(age_query)

with insight_col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Readmission Rate by Age Group</div>', unsafe_allow_html=True)

    fig_age = px.bar(
        age_df,
        x="age_group",
        y="readmission_rate",
        labels={"age_group": "Age Group", "readmission_rate": "Readmission Rate (%)"},
        text="readmission_rate",
        color_discrete_sequence=["#1a4fa3"],
    )
    fig_age.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="Age Group: %{x}<br>Readmission Rate: %{y:.2f}%<extra></extra>",
    )
    st.plotly_chart(style_fig(fig_age), use_container_width=True)

    with st.expander("View underlying data"):
        st.dataframe(age_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Readmission rate by diagnosis category ----------------------------------------
diagnosis_query = """
    SELECT
        diagnosis_category,
        ROUND(
            100.0 * SUM(readmitted_within_30_days) / COUNT(*),
            2
        ) AS readmission_rate
    FROM (
        SELECT DISTINCT
            d.encounter_id,
            d.diagnosis_category,
            r.readmitted_within_30_days
        FROM diagnoses d
        JOIN readmissions r
            ON d.encounter_id = r.encounter_id
        WHERE d.diagnosis_category IS NOT NULL
    ) AS diagnosis_encounters
    GROUP BY diagnosis_category
    ORDER BY readmission_rate DESC;
"""

diagnosis_df = run_query(diagnosis_query)

with insight_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Readmission Rate by Diagnosis Category</div>', unsafe_allow_html=True)

    # Sorted ascending here because horizontal bars read top-to-bottom;
    # this puts the highest readmission rate at the top of the chart.
    diagnosis_plot_df = diagnosis_df.sort_values("readmission_rate", ascending=True)

    fig_diagnosis = px.bar(
        diagnosis_plot_df,
        x="readmission_rate",
        y="diagnosis_category",
        orientation="h",
        labels={"diagnosis_category": "Diagnosis Category", "readmission_rate": "Readmission Rate (%)"},
        text="readmission_rate",
        color_discrete_sequence=["#1a4fa3"],
    )
    fig_diagnosis.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="Diagnosis: %{y}<br>Readmission Rate: %{x:.2f}%<extra></extra>",
    )
    st.plotly_chart(style_fig(fig_diagnosis), use_container_width=True)

    with st.expander("View underlying data"):
        st.dataframe(diagnosis_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Readmission rate by prior admission --------------------------------------------
prior_query = """
    SELECT
        CASE
            WHEN a.prior_admission = 1 THEN 'Prior admission'
            ELSE 'No prior admission'
        END AS prior_admission_group,
        COUNT(*) AS encounters,
        SUM(r.readmitted_within_30_days) AS readmitted_30_days,
        ROUND(
            100.0 * SUM(r.readmitted_within_30_days) / COUNT(*),
            2
        ) AS readmission_rate
    FROM admissions a
    JOIN readmissions r
        ON a.encounter_id = r.encounter_id
    GROUP BY a.prior_admission
    ORDER BY a.prior_admission DESC;
"""

prior_df = run_query(prior_query)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-subheading">Readmission Rate by Prior Admission History</div>', unsafe_allow_html=True)

fig_prior = px.bar(
    prior_df,
    x="prior_admission_group",
    y="readmission_rate",
    labels={"prior_admission_group": "Prior Admission History", "readmission_rate": "Readmission Rate (%)"},
    text="readmission_rate",
    color_discrete_sequence=["#1a4fa3"],
)
fig_prior.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    hovertemplate="%{x}<br>Readmission Rate: %{y:.2f}%<extra></extra>",
)
st.plotly_chart(style_fig(fig_prior), use_container_width=True)

with st.expander("View underlying data"):
    st.dataframe(prior_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Hospital Utilization
# ----------------------------------------------------------------------------
st.markdown('<div class="section-heading">Hospital Utilization</div>', unsafe_allow_html=True)

util_col1, util_col2 = st.columns(2)

# --- Length of stay by readmission status -------------------------------------------
los_query = """
    SELECT
        CASE
            WHEN r.readmitted_within_30_days = 1
                THEN 'Readmitted within 30 days'
            ELSE 'Not readmitted within 30 days'
        END AS readmission_group,
        COUNT(*) AS encounters,
        ROUND(AVG(a.length_of_stay_days), 2) AS avg_length_of_stay
    FROM admissions a
    JOIN readmissions r
        ON a.encounter_id = r.encounter_id
    GROUP BY r.readmitted_within_30_days
    ORDER BY r.readmitted_within_30_days DESC;
"""

los_df = run_query(los_query)

with util_col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Average Length of Stay by Readmission Status</div>', unsafe_allow_html=True)

    fig_los = px.bar(
        los_df,
        x="readmission_group",
        y="avg_length_of_stay",
        labels={"readmission_group": "Readmission Status", "avg_length_of_stay": "Avg. Length of Stay (days)"},
        text="avg_length_of_stay",
        color_discrete_sequence=["#1a4fa3"],
    )
    fig_los.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        hovertemplate="%{x}<br>Avg. Length of Stay: %{y:.2f} days<extra></extra>",
    )
    st.plotly_chart(style_fig(fig_los), use_container_width=True)

    with st.expander("View underlying data"):
        st.dataframe(los_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Admission type distribution ----------------------------------------------------
admission_query = """
    SELECT
        CASE admission_type_id
            WHEN 1 THEN 'Emergency'
            WHEN 2 THEN 'Urgent'
            WHEN 3 THEN 'Elective'
            WHEN 4 THEN 'Newborn'
            WHEN 5 THEN 'Not Available'
            WHEN 6 THEN 'Not Specified'
            WHEN 7 THEN 'Trauma Center'
            WHEN 8 THEN 'Not Mapped'
            ELSE 'Unknown'
        END AS admission_type,
        COUNT(*) AS encounters
    FROM admissions
    GROUP BY admission_type_id
    ORDER BY encounters DESC;
"""

admission_df = run_query(admission_query)

with util_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Admission Type Distribution</div>', unsafe_allow_html=True)

    admission_plot_df = admission_df.sort_values("encounters", ascending=True)

    fig_admission = px.bar(
        admission_plot_df,
        x="encounters",
        y="admission_type",
        orientation="h",
        labels={"admission_type": "Admission Type", "encounters": "Encounters"},
        text="encounters",
        color_discrete_sequence=["#1a4fa3"],
    )
    fig_admission.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="%{y}<br>Encounters: %{x:,}<extra></extra>",
    )
    st.plotly_chart(style_fig(fig_admission), use_container_width=True)

    with st.expander("View underlying data"):
        st.dataframe(admission_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Discharge disposition (full width, long category names) ------------------------
discharge_query = """
    SELECT
        discharge_disposition,
        COUNT(*) AS encounters
    FROM admissions
    GROUP BY discharge_disposition
    ORDER BY encounters DESC
    LIMIT 10;
"""

discharge_df = run_query(discharge_query)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-subheading">Top 10 Discharge Dispositions</div>', unsafe_allow_html=True)

discharge_plot_df = discharge_df.sort_values("encounters", ascending=True)

fig_discharge = px.bar(
    discharge_plot_df,
    x="encounters",
    y="discharge_disposition",
    orientation="h",
    labels={"discharge_disposition": "Discharge Disposition", "encounters": "Encounters"},
    text="encounters",
    color_discrete_sequence=["#1a4fa3"],
)
fig_discharge.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
    hovertemplate="%{y}<br>Encounters: %{x:,}<extra></extra>",
)
style_fig(fig_discharge)
fig_discharge.update_layout(height=420)  # taller than default, since this list has more categories
st.plotly_chart(fig_discharge, use_container_width=True)

with st.expander("View underlying data"):
    st.dataframe(discharge_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="dashboard-footer">Hospital Readmission Analytics | Academic Project | Part 1</div>',
    unsafe_allow_html=True,
)