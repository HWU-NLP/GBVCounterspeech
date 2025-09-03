import streamlit as st
import pandas as pd


# Fetch Prolific PID from study URL and store it in st.session_state for annotation task
def get_prolific_pid():
    """
    Fetch Prolific PID from study URL
    """
    query_params = st.query_params
    # st.write(query_params)
    if 'PROLIFIC_PID' in query_params:
        prolific_pid = query_params['PROLIFIC_PID']
        if isinstance(prolific_pid, list):  # Check if list and retrieve the first element
            return prolific_pid[0]
        return prolific_pid  # If it's already a string, return it directly
    return None

prolific_id = get_prolific_pid()
if prolific_id and 'PROLIFIC_PID' not in st.session_state:
    st.session_state.PROLIFIC_PID = prolific_id

# Create multiple pages
intro = st.Page("intro.py", title="🟢 Task Overview", default=True)
guideline = st.Page("guideline.py", title="🟠 Task Guideline")
task = st.Page("annotation_db.py", title="🔴 Start Annotation Task")


# Add pre-page (similiar to login) for information sheet and consent form before starting the annotation page
if "give_consent" not in st.session_state:
    st.session_state.give_consent = False

def give_consent():
    st.write("# Gender-Based Violence (GBV) Annotation Task")
    st.markdown("Thank you for your interest to participate in this task!")
    st.markdown("The following **Information Sheet** provides all the details about the task and how your data will be collected and stored.")

    st.error("**BEFORE YOU START**  \n Please read the information sheet carefully and click **YES** if you consent to participate.")

    st.markdown("### Information Sheet")
    st.markdown("**Study title:** Structured Gender-Based Violence and Counterspeech")
    st.markdown("**Investigators:** anonymised on acceptance.")

    st.markdown("🔹 **What are the possible risks of taking part in this study?**")
    st.markdown("Some of the examples you will see contain hateful language and may be offensive or upsetting.")
    st.markdown(":red[**WARNING** – this research exposes participants to offensive language – which includes hate speech or hateful language that may target people with disabilities, individuals of Jewish or Muslim faith, women, individuals who identify as LGBT+, people of colour, immigrants, and others – which may cause mental or physical stress to the reader. Please consider this before participating, you are under no obligation to take part and if you choose not to, we thank you for considering taking part. Please do remember that you can withdraw from participating at any time.]")
    
    st.markdown("🔹 **Has this research been approved?**")
    st.markdown("Yes. The project has been approved by the our institution ethics committee.") 
    
    st.markdown("🔹 **What is the purpose of this research?**")
    st.markdown("To research what strategies and targets are employed in countering online gender-based violence (GBV), and which are most effective for responding to gender-based abusive language. This research aims to help build and employ AI systems such as large language models that can better learn the relationship between structured gender-based hateful language and its counterspeech, and generate more effective counterspeech responses to mitigate the impact of GBV and foster more positive online interactions.")

    st.markdown("🔹 **Why have I been invited to take part?**")
    st.markdown("You have been invited to participate as you are a highly qualified worker on Prolific.")

    st.markdown("🔹 **Do I have to take part?**")
    st.markdown("Taking part is voluntary. You may choose not to take part or to decide to stop taking part at any time without giving a reason.")

    st.markdown("""
    🔹 **What do I have to do if I decide to participate?**
    
    Consent to participate in the study by checking 'Yes, I agree' in the consent form.
    
    If you participant in **<Gender-Based Violence (GBV) Annotation Task>**:
    1. Label a portion of the dataset that contains instances of gender-based hate speech.    
    2. Read each gender-based hateful text and determine the target, the target description, the intent and and write its statement.
    """)
    
    st.markdown("🔹 **What are the benefits of participating in this study?**")
    st.markdown("You will be paid at least the UK Living Wage per hour for participating in this part of the study.")
    st.markdown("You will be paid for your time regardless of whether you complete the study.")

    st.markdown("🔹 **What will happen if I don’t wish to carry on with the study?**")
    st.markdown("If you decide that you no longer wish to take part, you can withdraw at any time without giving a reason.")

    st.markdown("🔹 **Privacy and confidentiality**")
    st.markdown("Our institution is the data controller for the personal data collected in this project. This means that we are responsible under data protection law for making sure your personal information is kept secure, confidential and used only in the way you have been told it will be used.")
    st.markdown("Only the investigators will have access to the personal information that you provide.")
    st.markdown("No information that could identify you will appear in any report on the results of this research.")

    st.markdown("Should you wish to enquire about data protection, you can contact: xxx (anonymised on acceptance). For further information on data protection, see xxx (anonymised on acceptance).")
    st.markdown("We will collect and use your personal data for this project to undertake research in the public interest as part of our core purpose under our University Charter and Statutes.")
    st.markdown("If you decide to stop taking part in this study, we will not be able to withdraw all of your data from the study if this will have an adverse impact on the integrity and validity of the research.")
    st.markdown("If we can withdraw from the study your data that has been collected or generated within the research project, we will usually need to retain a record of your initial consent to participate within the project governance documentation for as long as required for audit purposes.")
    st.markdown("If you would like to know more about what our institution does with your personal data and your rights under privacy law, please visit our data protection web pages at xxx or contact our Data Protection Officer by email at xxx (anonymised on acceptance).")

    st.markdown("🔹 **What happens at the end of the project?**")
    st.markdown("The research team will use the findings of the research to produce a published report.")
    st.markdown("Please contact the researcher if you would like to obtain a copy of the report. We will not include any information that could identify you in any report or publication. We may keep data permanently where it is necessary for archiving purposes in the public interest.")

    st.markdown("🔹 **Will I be identifiable from my Prolific ID?**")
    st.markdown("**No.** Prolific fully anonymises all data.")

    st.markdown("🔹 **What if I have questions or concerns?**")
    st.markdown("If you have questions about any aspect of this study, please contact xxx (anonymised on acceptance)")
    st.markdown("If you have a concern or complaint about the study, please contact xxx (anonymised on acceptance)")

    st.markdown("**Please read the text in red above carefully before you consider participating in this study, as you may be asked to label samples of hateful/offensive speech.**")
    st.markdown("If you have any questions or concerns about the use of your personal data or your rights under data protection law, please contact xxx (anonymised on acceptance).")
    st.markdown("Thank you for reading this information sheet and for your interest in this research. ")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""### Consent
    I have read and understand the Information Sheet.

    I understand the expected potential risks to me in my participation.

    I understand that this research does expose participants to offensive language,
    including hate speech directed at various groups. This includes hateful language that
    may target people with disabilities, individuals of Jewish or Muslim faith, women,
    individuals who identify as LGBT+, people of colour, immigrants, and others, which
    may cause mental or physical stress to the reader.

    I understand that I can refuse to answer any questions or stop taking part at any time without
    any of my rights being affected.

    I understand that my data may be kept indefinitely for use in future ethically approved
    research and may be used in future ethically approved research and archived in the public
    interest.

    I do not have any physical disabilities, mental health issues, or other conditions that might
    cause me to be negatively impacted me by participating in this research study.

    Please select YES or NO to state whether you agree:

    I agree to take part in this study""")


    col1 = st.button("YES")
    col2 = st.button("NO")
 
    if col1 and not col2:
        st.session_state.give_consent = True
        st.rerun()


    elif col2 and not col1:
        st.error("Sorry, you must read the information sheet and give the consent before you start our annotation task.")


# Integrate multiple pages
before_task_page = st.Page(give_consent, title="Before the task", icon=":material/login:")

if st.session_state.give_consent:
    pg = st.navigation([intro, guideline, task])
else:
    pg = st.navigation([before_task_page])

pg.run()
