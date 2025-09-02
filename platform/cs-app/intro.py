import streamlit as st
import pandas as pd

st.write("# GBV Counterspeech Annotation Task")

with st.sidebar:
    st.markdown("""
    **IMPORTANT NOTES**
    \n\n 👉 The **full annotation guideline** is provided below. Please **read carefully and pass a qualification test** before starting. 
    \n\n 👉 This task has a maximum participant limit, so the task may **close earlier** if some participants submit multiple entries and our data annotation target is reached. Please follow the instruction or report to us on Prolific if you meet the issue **NO DATA AVAILABLE**.
    \n\n 👉 If you wish to exit our task before completion, click the button **Exit GBV Counterspeech Annotation** in the sidebar to confirm your participation.
    \n\n 👉 Regardless of whether you complete the study—whether you spent time reading the guidelines but failed the qualification test or completed some annotations but not all—you will still **receive a bonus** for your time and effort. However, you may be **asked to return the task** on Prolific. Your work will be manually reviewed, and your bonus will be determined accordingly.
    """)
    st.markdown("<br>", unsafe_allow_html=True)


st.success("Welcome to our study for GBV Counterspeech Annotation Task! 👋")
st.markdown("In this task, you will label examples of :blue-background[**counterspeech**]. Counterspeech is essentially a response written by experts from charities to counter **gender-based violence (GBV) hatespeech** texts.")
st.markdown("There is one task to be annotated: :blue-background[Strategy]. **Feedback** is also welcome after the annotation.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🟠 HOW TO START")
st.markdown("""
1. Read the annotation guidelines carefully by clicking **Task Guideline** in the sidebar to go to the task guideline page.
2. Move to the annotation task page by clicking **Start Annotation Task** in the sidebar.  
2. Complete the **qualification test** to qualify. You **must pass the test** before moving on to the formal annotation task.  
3. After passing the qualification test and when data samples are assigned to you, start annotating by completing the task and providing your feedback.  
""")
st.markdown("<br>", unsafe_allow_html=True)

