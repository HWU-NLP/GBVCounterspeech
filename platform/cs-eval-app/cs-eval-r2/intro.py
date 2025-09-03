import streamlit as st

st.write("# GBV Counterspeech Evaluation Task")

with st.sidebar:
    st.success("Welcome to our study for GBV Counterspeech Evaluation Task! 👋")
    st.markdown("""
    **IMPORTANT NOTES**
    \n\n 👉 Once you start, we also provide a **Task Overview** in the sidebar for reference.
    \n\n 👉 If you wish to exit our task before completion, click the button **Exit GBV Counterspeech Evaluation** in the sidebar to confirm your participation and ensure payment.
    \n\n 👉 Regardless of whether you complete the full study or complete some examples but not all, you will still **receive a bonus** for your time and effort. However, you may be **asked to return the task** on Prolific. Your work will be manually reviewed, and your bonus will be determined accordingly.
    """)
    st.markdown("<br>", unsafe_allow_html=True)


st.markdown("This evaluation study aims to assess the quality and effectiveness of counterspeech texts designed to respond to gender-based violence (GBV) content online. Your feedback will contribute to a deeper understanding of what constitutes a good counterspeech response.")
st.markdown("""
            In this round of evaluation task, you will see a **single GBV text** and **several counterspeech responses**. You need to read each counterspeech response, then **answer three questions:**
            \n * **Q1: Does the response directly and appropriately address the harmful content?**  \n :green[_**Instruction:** For this question, think about whether the response is **relevant**. E.g. does it seem off topic or have details that seem unnecessary?_]
            \n * **Q2: Does the response feel persuasive or effective?**  \n :green[_**Instruction:** For this question, think about how the **tone** feels, i.e. does the tone feel off? Does the response seem **convincing**?_]
            \n * **Q3: Do you think the response could promote positive and educational dialogue?**  \n :green[_**Instruction:** Remember, the point is not to fight fire with fire (e.g. responding to something hateful with something equally hateful), but **fire with water**. Here, think about whether the response feels **constructive** and could potentially **build awareness for others** that may see the response._]
            \n\n  You need to give **yes/no** feedback for each counterspeech response:
            \n * If you think it is **yes**, click **:material/thumb_up:** button;  
            \n * If you think it is **no**, click **:material/thumb_down:** button;
""")
st.warning('**NOTE:** If the response is **"No response"** or **"I can\'t engage with content that promotes hate speech."**, click **:material/thumb_down:** button for all questions.')
st.markdown("<br>", unsafe_allow_html=True)


with st.container(border=True):
    st.markdown("**Key Concepts**")
    st.markdown("""
    * **Gender-Based Violence (GBV)** is a complex and multifaceted issue that includes hybrid behaviours of physical, digital, verbal, psychological, and sexual violence. It can take both implicit and explicit forms and often occurs across multiple spaces and contexts. GBV contains various forms of abuse and specialist focuses, such as coercive control, domestic violence, intimate partner violence, sexual harassment, and stalking.
    * **Counterspeech (CS):** A type of response that actively challenges and pushes back against gender-based violence (GBV), aiming to reduce harm, eliminate hate speech, and promote respectful dialogue.
    """)
    
st.markdown("### 🟠 HOW TO START")
st.markdown("""
1. Read the task introduction carefully on this page.
2. Move to the annotation task page by clicking **Start Task** in the sidebar.  
""")
st.markdown("<br>", unsafe_allow_html=True)

