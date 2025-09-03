import streamlit as st

st.write("# Evaluating Responses to Gender-Based Violence (GBV) Posts")

with st.sidebar:
    st.success("Welcome to our study for Evaluating Responses to Gender-Based Violence Posts! 👋")
    st.markdown("""
    **IMPORTANT NOTES**
    \n\n 👉 Once you start, we also provide a **Task Overview** in the sidebar for reference.
    \n\n 👉 If you wish to exit our task before completion, click the button **Exit GBV Counterspeech Evaluation** in the sidebar to confirm your participation and ensure payment.
    \n\n 👉 Regardless of whether you complete the full study or complete some examples but not all, you will still **receive a bonus** for your time and effort. However, you may be **asked to return the task** on Prolific. Your work will be manually reviewed, and your bonus will be determined accordingly.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("In this task, remember that you are **placed in a more realistic scenario** similar to what users may encounter on social media platforms.")
st.markdown("In each example, you will see a **single GBV post** and **4 responses**. These include **two counterspeech responses** and **two non-counterspeech alternatives**. Read the explanations of the different responses below, and **select one or more responses** you would prefer for this GBV post.")
st.markdown("""
* **Response 1 & 2: counterspeech responses**
\nCounterspeech aims to challenge the harmful message, promote constructive dialogue and reduce harm. However, some responses might unintentionally worsen the situation or make bystanders feel uncomfortable, depending on the tone and context.
* **Response 3: content moderation-style message**
\nThis kind of warning is commonly used on social platforms like Reddit, YouTube, and other community-guided platforms. It flags the harmful post that breaks community rules and discourages further engagement. This approach may reduce harm, but it avoids directly confronting the content.
\n:green[_E.g.  “This post has been flagged for violating community guidelines on gender-based violence. Continued engagement in this thread may lead to content being filtered or restricted.”_]
* **Response 4: disengagement option**
\nThis option allows you to disengage from the conversation entirely. It is often a valid and self-protective decision, particularly in triggering or unsafe contexts. However, disengagement may leave the original harm to persist unaddressed, or may miss the opportunity to support others affected by the content.
\n:green[_E.g. “I do not wish to engage in this conversation.”_]
""")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
You need to **give your selection** for each response:
\n * If you **like** this response, click **:material/thumb_up:** button;  
\n * If you **dislike** this response, click **:material/thumb_down:** button;
""")

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**Gender-Based Violence (GBV) Definition**")
    st.markdown("""
    It is a complex and multifaceted issue that includes hybrid behaviours of physical, digital, verbal, psychological, and sexual violence. It can take both implicit and explicit forms and often occurs across multiple spaces and contexts. GBV contains various forms of abuse and specialist focuses, such as coercive control, domestic violence, intimate partner violence, sexual harassment, and stalking.
    """)
    
st.markdown("### 🟠 HOW TO START")
st.markdown("""
1. Read the task introduction carefully on this page.
2. Move to the annotation task page by clicking **Start Task** in the sidebar.  
""")
st.markdown("<br>", unsafe_allow_html=True)

