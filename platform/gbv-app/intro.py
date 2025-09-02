import streamlit as st
import pandas as pd

st.write("# Gender-Based Violence (GBV) Annotation Task")

with st.sidebar:
    st.warning("""
    **IMPORTANT NOTES**
    \n\n 👉 The **full annotation guideline** is provided in the guideline page. Please **read carefully and pass a qualification test** before starting.  
    \n\n 👉 This task has a maximum participant limit, so the task may **close earlier** if some participants submit multiple entries and our data annotation target is reached. Please follow the instruction or report to us on Prolific if you meet the issue **NO DATA AVAILABLE**.
    \n\n 👉 If you wish to exit our task before completion, click the button **Exit GBV Annotation Task** found in the sidebar in the annotation task page to confirm your participation.
    \n\n 👉 Regardless of whether you complete the study—whether you spent time reading the guidelines but failed the qualification test or completed some annotations but not all—you will still **receive a bonus** for your time and effort. However, you may be **asked to return the task** on Prolific. Your work will be manually reviewed, and your bonus will be determined accordingly.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

st.success("Welcome to our study for Gender-Based Violence (GBV) Annotation Task! 👋")
st.markdown(f"This task aims to tackle online gender-based hate speech, by providing different perspectives via response. In this task, you will take a look at :blue-background[**hateful texts related to gender-based violence**] that are collected from social media platforms.")
st.markdown("There are **two subtasks** to be annotated: :blue-background[GBV Target, and GBV Form]. **Feedback** is also welcome after the annotation.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🟠 HOW TO START")
st.markdown("""
1. Read the annotation guidelines carefully by clicking **Task Guideline** in the sidebar to go to the task guideline page.
2. Move to the annotation task page by clicking **Start Annotation Task** in the sidebar.  
2. Complete the **qualification test** to qualify. You **must pass the test** before moving on to the formal annotation task.  
3. After passing the qualification test and when data samples are assigned to you, start annotating by completing the three subtasks and providing your feedback.  
""")








