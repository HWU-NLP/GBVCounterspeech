import random
import os
from pathlib import Path
import pandas as pd
import time
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import (
    init_database, 
    SourceDataset, 
    SessionRecord, 
    AnnotationRecord, 
    PairCompletion, 
    AnnotatorInfo, 
    Assignment
)
from sqlalchemy import or_, not_, func
from text_highlighter import text_highlighter
import re

TOTAL_CASES = 30  # number of pairs assigned per participant, defaul=30 per 30 mins
ITEMS_PER_PAGE = 1  # only display 1 case per page
NUM_ANNOTATOR_PER_ITEM = 1  # default=3 for maximum annotator per item, add 1 for tolerance

CUSTOMISED_USER_ID = "123456abc"

# Prolific completion code - TODO: check code every time before deployment
COMPLETION_CODE = "C153LHOR"
COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C153LHOR"
STOP_BEFORE_TASK_CODE = "CBDKFXNF"
STOP_BEFORE_TASK_URL = "https://app.prolific.com/submissions/complete?cc=CBDKFXNF"
SCREEN_OUT_CODE = "CN97TWGC"  
SCREEN_OUT_URL = "https://app.prolific.com/submissions/complete?cc=CN97TWGC"
NO_DATA_CODE = "CPS95BSW"
NO_DATA_URL = "https://app.prolific.com/submissions/complete?cc=CPS95BSW"

# Create the SQLite engine and session
DATABASE_URL = "sqlite:///gbv_anno.db"  # TODO: rename it based on annotation round e.g. pilot_db
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Initialise session state for guidelines toggle, page setting
if 'page' not in st.session_state:
    st.session_state.page = -1  # -1 for test pair, 0 for formal pairs
if 'subset' not in st.session_state:
    st.session_state.subset = [False]
if 'exit_clicked1' not in st.session_state:
    st.session_state.exit_clicked1 = False
if 'exit_clicked2' not in st.session_state:
    st.session_state.exit_clicked2 = False
if 'submit_and_correct' not in st.session_state:
    st.session_state.submit_and_correct = False
if 'submit_but_wrong' not in st.session_state:
    st.session_state.submit_but_wrong = False
if 'start_task' not in st.session_state:
    st.session_state.start_task = False
if 'page_clicked' not in st.session_state:
    st.session_state.page_clicked = False
if 'DEPLOY' not in st.session_state:
    st.session_state.DEPLOY = False  # DEPLOY mode for hosting on Prolific

# Initialise the database with dataset
init_database('data/gbv_sample.tsv', engine)

# Instantiate a database session
session = SessionLocal()

# gbv form
gbv_forms = [
    "Dehumanisation",
    "Threatening",
    "Derogation",
    "Animosity",
    "Support of Hate Crimes",
]


# qualification questions for Section 1 - concept questions
concept_questions = [
    {"question": "Gender-based violence (GBV) includes only physical violence.", "answer": False},
    {"question": "A single GBV text can have multiple GBV forms.", "answer": True},
    {"question": "If a text does not include explicit gender-based insults, it cannot be GBV.", "answer": False},
    {"question": "Threatening statements always include explicit calls for harm.", "answer": False},
    {"question": "GBV can target not only individuals/groups (e.g., women) but also ideologies, such as feminism and gender equality.", "answer": True},
]

multiple_choice_questions = [
    {"question": "What is the expected output for this annotation task?",
     "options": ["Identify the text contains GBV content or not", "Highlight the content related to GBV target", "Label gbv forms", "Highlight key GBV content"],
     "correct": ["Highlight the content related to GBV target", "Label gbv forms"]},
    {"question": "Which of the following is NOT a GBV form?",
     "options": ["Dehumanisation", "Encouragement", "Denouncing", "Threatening"],
     "correct": ["Encouragement", "Denouncing"]},
    {"question": "What is the key difference between Animosity and Derogation? (Single choice questions)",
     "options": ["Animosity is explicit insults, while Derogation is implicit hostility.", "Animosity includes implicit or subtle hostility, while Derogation involves explicit insults or demeaning language.", "Animosity and Derogation are the same and can be used interchangeably.", "Animosity is more aggressive than Derogation and always includes threats."],
     "correct": ["Animosity includes implicit or subtle hostility, while Derogation involves explicit insults or demeaning language."]},
]


# qualification test - annotation practice samples
annotation_samples = [
    {"gbv": "Women's football is so slow and clumsy.",
     "correct_target": "Women",
     "correct_forms": "Derogation",},
    {"gbv": "Women are pigs.",
     "correct_target": "Women",
     "correct_forms": "Dehumanisation",},
]

def get_user_id():
    """
    Get user id from Prolific platform
    """
    if st.session_state.DEPLOY:
        user_id = st.session_state.PROLIFIC_PID
    else:
        user_id = CUSTOMISED_USER_ID  # set a default user id   
    
    if not user_id:
        st.error("No Prolific PID found. Please make sure you are accessing the app through the Prolific platform.")
    return user_id

def get_user_action(user_id, session):
    """
    Check user action history in SessionRecord.
    """
    user_action = session.query(SessionRecord).filter_by(user_id=user_id).first()
    return user_action

def update_user_test_info(user_id, session):
    """
    Update the status of passing the test annotation or not for a user in AnnotatorInfo.
    """
    user_info = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
    if not user_info:
        user_info = AnnotatorInfo(user_id=user_id, num_text=0, passed=1)
        session.add(user_info)
    else:
        user_info.passed = 1
    session.commit()
    
def update_user_anno_info(user_id, pair_id, session):
    """
    Update the number of annotated text for a user in AnnotatorInfo.
    """
    user_info = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
    if not user_info:
        user_info = AnnotatorInfo(user_id=user_id, num_text=1, passed=1, pairs=pair_id)
        session.add(user_info)
    else:
        user_info.num_text += 1
        
        # record a list of completed pairs for each user -> avoid the case that annnotation of one pair is done and saved in num_complete, but the user refresh the page and click save button again to increase the num_complete
        if user_info.pairs:
            user_info.pairs += f", {pair_id}"  
        else:
            user_info.pairs = pair_id
    session.commit()

def update_pair_assignment(pair_id, assign_id, session, passed=True):
    """
    Update the number of assignments for a pair in PairCompletion.
    Update the assignment record for a pair.
    """
    if pair_id is None:
        raise ValueError("[ERROR] pair_id is None! This will cause an IntegrityError.")

    pair_status = session.query(PairCompletion).filter_by(pair_id=pair_id).first()
    if not pair_status:
        session.add(PairCompletion(pair_id=pair_id, num_assign=1, num_complete=0))
        # session.add(Assignment(pair_id=pair_id, assign_id=assign_id))
    
    else:
        pair_status.num_assign += 1
    session.commit()

def update_pair_completion(pair_id, session):
    """
    Update the number of completions for a pair in PairCompletion.
    """
    completion = session.query(PairCompletion).filter_by(pair_id=pair_id).first()
    if not completion:
        completion = PairCompletion(pair_id=pair_id, num_assign=1, num_complete=1)
        session.add(completion)
    else:
        completion.num_complete += 1
    session.commit()

def assign_new_subset(user_id, assign_id, session):
    """
    Assign a new subset of pairs to the user.
    """
    pairs = session.query(SourceDataset).all()
    
    subset = []
    selected_pair_ids = []
    checked_pair_ids = set() 
    while len(subset) < TOTAL_CASES: # and num_assignments < total_pairs:
        # exit loop if all pairs have been checked
        if len(checked_pair_ids) == len(pairs):
            break

        pair = random.choice(pairs)

        # skip if already checked
        if pair.pair_id in checked_pair_ids:
            continue
        
        checked_pair_ids.add(pair.pair_id) 

        # skip duplicate pairs
        if pair.pair_id in selected_pair_ids:
            continue

        # skip pairs assigned by required number of annotators
        completion = session.query(PairCompletion).filter_by(pair_id=pair.pair_id).first()
        if completion:
            if completion.num_complete >= NUM_ANNOTATOR_PER_ITEM or completion.num_assign >= NUM_ANNOTATOR_PER_ITEM:
                continue
            
        # update_pair_assignment(pair.pair_id, session)

        existing_annotation = session.query(AnnotationRecord).filter_by(
            user_id=user_id, pair_id=pair.pair_id
        ).first()

        if not existing_annotation:
            subset.append((pair.pair_id, "", "", "")) 
            selected_pair_ids.append(pair.pair_id)    
        elif any(
            getattr(existing_annotation, col) == ""
            for col in ["gbv_target", "gbv_form", "feedback"]
        ):
            subset.append((pair.pair_id, "", "", ""))
            selected_pair_ids.append(pair.pair_id)

    # Update assignments and annotation records
    for item in subset:
        pair_id = item[0]
        update_pair_assignment(pair_id, assign_id, session)
        session.add(Assignment(pair_id=pair_id, assign_id=assign_id))

        # Check if the record exists before adding
        existing_annotation = session.query(AnnotationRecord).filter_by(
            user_id=user_id, pair_id=pair_id
        ).first()
        if not existing_annotation:
            session.add(AnnotationRecord(user_id=user_id, 
                                        pair_id=pair_id, 
                                        gbv_target = "", 
                                        gbv_form = "", 
                                        feedback="",
                                        ))
        else:
            # Update the label if needed (optional)
            existing_annotation.gbv_target = ""
            existing_annotation.gbv_form = ""
            existing_annotation.feedback = ""

    session.commit()
    return subset

def update_annotation_record(user_id, pair_id, labels, session):
    """
    Update annotation results to database per pair and update pair completion.
    """
    annotation = session.query(AnnotationRecord).filter_by(
            user_id=user_id, pair_id=pair_id
        ).first()
    if annotation:
        annotation.gbv_target = labels[0]
        annotation.gbv_form = labels[1]
        annotation.feedback = labels[2]
    else:
        session.add(AnnotationRecord(
            user_id=user_id, 
            pair_id=pair_id, 
            gbv_target = labels[0],
            gbv_form = labels[1], 
            feedback = labels[2]
        ))
    # update_pair_completion(pair_id, session)
    # update_user_info(user_id, session)
    session.commit()


# Three scenarios for a user:
# 1. a new annotator start the task
# 2. a recorded annotator return to the task again and previous task has been completed
# 3. a recorded annotator return to the task again but previous task has not been completed / refresh
def handle_scenario_1(user_id, session):
    """
    Handle new annotator scenario.
    """
    assign_id = f"{user_id}-1"
    
    subset = assign_new_subset(user_id, assign_id, session)
    record = SessionRecord(user_id=user_id, assign_id=assign_id, status=1, time=0)

    session.add(record)
    session.add(AnnotatorInfo(user_id=user_id))
    session.commit()
    
    return subset, record

def handle_scenario_2(user_id, record, session):
    """
    Handle recorded annotator with completed task scenario.
    """
    round_num = int(record.assign_id.split('-')[1]) + 1
    assign_id = f"{user_id}-{round_num}"

    record.assign_id = assign_id
    record.status = 1
    session.commit()

    subset = assign_new_subset(user_id, assign_id, session)
    return subset, record


def handle_scenario_3(user_id, record, session):
    """
    Handle recorded annotator with incomplete task scenario.
    """
    assign_id = record.assign_id

    # Retrieve all pairs assigned in the previous session
    assignments = session.query(Assignment).filter_by(assign_id=assign_id).all()
    subset = []
    for assignment in assignments:
        annotation = session.query(AnnotationRecord).filter_by(
            user_id=user_id, pair_id=assignment.pair_id
        ).first()

        if annotation:
            # If annotation exists, keep the existing label
            subset.append((assignment.pair_id, 
                        annotation.gbv_target, 
                        annotation.gbv_form,
                        annotation.feedback,
                        )) 

        else:
            # If annotation does not exist, leave it empty
            subset.append((assignment.pair_id, "", "", ""))  

    return subset, record

def get_max_button_size(gbv_forms):
    """
    Calculate the max width and height for buttons
    """
    max_width = max([len(form) for form in gbv_forms]) * 10  # Estimate width
    max_height = 60  # Fixed height
    return max_width, max_height

def show_progress_bar(user_id, subset_size, session):
    """
    Show progress bar for annotation task per page/example
    """
    # Check how many pairs in the current page are completed for 4 subtasks
    user_action = session.query(SessionRecord).filter_by(user_id=user_id).first()
    num_completed_pairs = session.query(AnnotationRecord).filter(
        AnnotationRecord.user_id == user_id,
        AnnotationRecord.pair_id.in_([assignment.pair_id 
                                    for assignment in session.query(Assignment).filter_by(assign_id=user_action.assign_id)])
    ).filter(
        or_(
            not_(AnnotationRecord.gbv_target == ""),
            not_(AnnotationRecord.gbv_form == ""),
            not_(AnnotationRecord.feedback == ""),
        )
    ).count()

    # Calculate progress for progress bar
    progress = num_completed_pairs / subset_size #TOTAL_CASES
    
    # display progress bar and progress text
    st.markdown("<br>", unsafe_allow_html=True)
    progress_text = f"⏳ Annotation Progress: {num_completed_pairs}/{subset_size}"
    st.progress(progress, text=progress_text)

def find_indices(text, substring):
    """
    Find the start and end indices of a substring in a text.
    """
    start = text.find(substring)
    if start == -1:
        return None  # Substring not found
    end = start + len(substring)  # End index (exclusive)
    return start, end
                   
def main():
    ### Custom CSS to ensure all buttons have the same height
    max_width, max_height = get_max_button_size(gbv_forms)

    st.markdown(
        f"""
        <style>
            div[data-testid="stHorizontalBlock"] > div {{
                flex: 1;
                max-width: {max_width}px;
                max-height: {max_height}px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            button {{
                width: 100%;
                height: {max_height/2}px;
                font-size: 16px;
                margin: 0px;
            }}
        </style>
        """, 
        unsafe_allow_html=True
    )

    # ### Task title
    # st.title("Gender-Based Violence (GBV) Annotation Task")

    ### Load Prolific ID and assign pairs to users based on 3 scenarios
    start_time = time.time()  # Start the timer
    user_id = get_user_id()
    
    
    ### Display a pair text for testing before starting annotation 
    # -> if page == -1 && no passed record of test in AnnotatorInfo
    # -> else load from the first incompleted pair
    
    if st.session_state.page == -1: 
        
        # Create sidebar with exit button for participants who stopped before formal task
        with st.sidebar:
            st.warning("**⏹️ EXIT BEFORE FORMAL ANNOTATION**  \nIf you have partially completed the task but wish to exit, click the button below to confirm your participation and exit.")
            exit1 = st.button("Exit GBV Annotation Task")    
            if exit1:
                st.session_state.exit_clicked1 = True

                st.success(f"Thank you for participating in the task! Please click **[here]({STOP_BEFORE_TASK_URL})** to confirm your participation on Prolific.  \n\n Please **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("You may now close this window!")
                st.stop()
            
            
        user_info = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
        
        if not user_info or not user_info.passed:
            ### Task title
            st.title("GBV Annotation Qualification Test")
        
            st.markdown(f"The qualification task consists of two sections: **Concept Understanding** and **Annotation Practice**.")
            st.markdown(f"**GOAL:** Ensure annotators understand key concepts and tasks before starting annotation. The test is automatically graded, and annotators must achieve :red[**at least 80% accuracy**] to qualify.")
            
            st.subheader("Section 1 - Concept Understanding")
            s1_score = 0
            s1_total = len(concept_questions) + len(multiple_choice_questions)

            ## True/False Questions
            st.markdown("##### 🟢 (I) True/False Questions")
            for idx, q in enumerate(concept_questions):
                response_TF = st.radio(label=f"**{idx+1}. {q['question']}**", 
                                       options=("True", "False"), 
                                       index=None, 
                                       key=f"{user_id}_TF_{idx+1}") #, label_visibility="collapsed")
                
                if response_TF is not None:  # Ensure user selects an answer
                    if (response_TF == "True" and q["answer"]) or (response_TF == "False" and not q["answer"]):
                        s1_score += 1
            
            st.markdown("<br>", unsafe_allow_html=True)

            ## Multiple-Choice Questions
            st.markdown("##### 🟢 (II) Multiple-Choice Questions")
            
            # Add partial score to total score for multiple choice questions
            for idx, q in enumerate(multiple_choice_questions):
                response_MCQ = st.multiselect(label=f"**{idx+6}. {q['question']}**", 
                                              options=q["options"], 
                                              key=f"{user_id}_MCQ_{idx+6}")
                
                correct_set = set(q["correct"])
                selected_set = set(response_MCQ)
                
                correct_selections = len(selected_set & correct_set)  # Intersection (correctly selected)
                incorrect_selections = len(selected_set - correct_set)  # Extra incorrect choices
                total_correct = len(correct_set)
                total_options = len(q["options"])  # Total answer choices

                # Calculate partial credit score
                partial_score = max(0, (correct_selections / total_correct) - (incorrect_selections / total_options))

                s1_score += partial_score  
            st.markdown("<br>", unsafe_allow_html=True)
            
            
            st.subheader("Section 2 - Annotation Practice")
            s2_score = 0
            s2_total = len(annotation_samples) * 2
            
            def check_answers_from_highlighter(gbv, selected_text, correct_answer):
                """
                Check if the selected text is a reasonable extraction of the GBV text.
                """

                max_length = int(len(gbv) * 0.8)  # 80% of the length of the GBV text
                selected_text = selected_text.lower().strip()

                # overlap check: does selection contain expected key phrases?
                overlap_valid = any(phrase.lower() in selected_text for phrase in correct_answer)
                
                # length check: Ensure selection is not too long
                length_valid = len(selected_text) <= max_length

                if overlap_valid and length_valid:
                    return True
                else:
                    return False

            
            for idx, sample in enumerate(annotation_samples):
                with st.container(border=True):
                    st.success(f"**TEST GBV EXAMPLE {idx + 1}**  \n\n {sample['gbv']}")
                    st.markdown("<br>", unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["**SUBTASK 1 - GBV Target**", "**SUBTASK 2 - GBV Form**"])

                    with tab1:
                        st.markdown(
                            "<p style='font-size:14px; color:grey;'>" 
                            "Highlight the particular sub-category used to refer to the <b>woman/women or ideology being targeted</b> "
                            "(e.g. a breastfeeding woman, a girl, feminist, feminism, gender equality, etc.). "
                            "Select the text <b>from left to right</b> to apply the colour shown above, and <b>double-click</b> to remove the selection."
                            "</p>",
                            unsafe_allow_html=True
                        )
                        gbv_target_selected = text_highlighter(
                            text=sample['gbv'],
                            labels=[
                                ("GBV target", "lightgreen"),
                            ],
                        )
                        
                        if gbv_target_selected:
                            gbv_target_input = sample['gbv'][gbv_target_selected[0]["start"]:gbv_target_selected[0]["end"]]
                        else:
                            gbv_target_input = ""
                            
                        if check_answers_from_highlighter(sample['gbv'], gbv_target_input, sample["correct_target"]):
                            s2_score += 1
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    with tab2:
                        selected_test_form = st.multiselect(label=":grey[Select up to 2 forms for GBV text]",
                                                    options=gbv_forms,
                                                    key=f"form_{idx}_{user_id}")
                        
                        if sample["correct_forms"] in selected_test_form and len(selected_test_form) <= 2:
                            s2_score += 1
                        st.markdown("<br>", unsafe_allow_html=True)
          
                st.markdown("<br>", unsafe_allow_html=True)
                
            
            ## Final Qualification Check
            total_score = s1_score + s2_score
            total_possible = s1_total + s2_total
            accuracy = (total_score / total_possible) * 100
            
            ### Design action buttons for moving to the next step
            col1, _, col3 = st.columns([1, 6, 1])
            with col1:
                submit = st.button("Submit Answers")   
            with col3:
                start = st.button("Start Annotation")  
            
            if not st.session_state.submit_and_correct and not st.session_state.submit_but_wrong and not st.session_state.start_task:
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("⚠️ Please complete the qualification tasks and click **Submit Answers** to qualify. If passed, you can then click **Start Annotation** to move to formal annotation task.")
            
            if submit:
                if accuracy >= 80:
                    st.session_state.submit_and_correct = True
                    st.session_state.submit_but_wrong = False
                    st.rerun()
                else:
                    st.session_state.submit_and_correct = False
                    st.session_state.submit_but_wrong = True
                    st.rerun()
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success("🎉 **Congratulations!**  \n You have passed the test. Please click **Start Annotation** button to start the annotation.")
                    st.session_state.submit_and_correct = True
            
            if st.session_state.submit_but_wrong and not start:
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Your total score: {accuracy:.1f}**")
                st.error("❌ Unfortunately, you did not pass.  \n Please review the guidelines and examples carefully, and try to submit your answers again.")
                st.markdown("If you do not want to continue the task, please click the **Exit GBV Annotation** button in the sidebar to confirm your participation.")
              
            if st.session_state.submit_and_correct and not start:
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Your total score: {accuracy:.1f}**")
                st.success("🎉 **Congratulations!**  \n You have passed the test. Please click **Start Annotation** button to start the annotation.")
            
            if start:
                if st.session_state.submit_and_correct:
                    st.session_state.start_task = True
                    st.session_state.page = 0
                    st.rerun()
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.warning("⚠️ Please complete the qualification tasks and submit answers to qualify before starting the formal annotation.")
        
        else:  
            if user_info:
                completed_pairs = user_info.pairs.split(", ")
                st.session_state.page = len(completed_pairs) # re-start task from the last completed pair
            else:
                st.session_state.page = 0
                
            st.session_state.start_task = True
            st.rerun()
    
       
    if st.session_state.start_task and st.session_state.page >= 0: 
        record = get_user_action(user_id, session)
        if not record:
            subset, record = handle_scenario_1(user_id, session)
        elif record.status == 0:
            subset, record = handle_scenario_2(user_id, record, session)
        elif record.status == 1:
            subset, record = handle_scenario_3(user_id, record, session)
        else:
            st.write("No data has been assigned.") 

        st.session_state.subset = subset
        st.session_state.start_task = False
        update_user_test_info(user_id, session)
        st.rerun()
    
    #================================================================================================
    ### Start annotation
    
    subset_size = len(st.session_state.subset)
    
    if st.session_state.page >= 0:
        ### Create sidebar to present annotation guideline
        with st.sidebar:
            st.warning("**⏹️ EXIT BEFORE COMPLETION**  \nIf you have partially completed the annotations but wish to exit the task, click the button below to confirm your participation and exit.")
            exit2 = st.button("Exit GBV Annotation Task")    
            if exit2:
                st.session_state.exit_clicked2 = True
                st.success(f"Thank you for participating in the task! Please click **[here]({SCREEN_OUT_URL})** to confirm your participation on Prolific.  \n\n Please **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("You may now close this window!")
                st.stop()

            st.markdown(f"------")
            st.subheader("🟢 Task Overview of GBV Annotation")
            st.markdown("For each GBV text, two subtasks need to be annotated: GBV Target, and GBV Form.")
            st.markdown("* **Subtask 1 - GBV Target:** Is there a particular sub-category used to refer to the **woman/women or ideology being targeted**? Please highlight the relevant text.  \n _It could be 'a breastfeeding woman, a girl, feminist, feminism, gender equality, etc.'. E.g. **Gender equality** is just a war on masculinity._")
            st.markdown("* **Subtask 2 - GBV Form:** The type of GBV expressed in the text. Here are five options: Dehumanisation, Threatening, Derogation, Animosity, and Support of Hate Crimes. You may select **up to two forms** if needed.")
            st.markdown("**Feedback** is also welcome after annotating each text to see if any issues you meet during the annotation process.")

            st.markdown(f"------")
            st.subheader("🟢 GBV Annotation Guidelines")
            st.markdown("<br>", unsafe_allow_html=True)
            
            show_gbv_guideline1 = st.toggle("View GBV Form Annotation Guideline")
            if show_gbv_guideline1:
            # Show or hide the annotation guidelines based on user selection
                st.markdown("#### GBV Form Annotation Guideline")
                st.markdown(
                    """
                    <style>
                        table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        th, td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #f4f4f4;
                            font-weight: bold;
                        }
                        .bold { font-weight: bold; }
                        .italic { font-style: italic; }
                        .underline { text-decoration: underline; }
                        .yellow { color: #D4A017; } /* Golden yellow for tone label */
                    </style>

                    <table>
                        <colgroup>
                            <col style="width: 20%;">
                            <col style="width: 80%;">
                        </colgroup>
                        <tr>
                            <th>Form</th>
                            <th>Definition</th>
                        </tr>
                        <tr>
                            <td><b>Dehumanisation</b></td>
                            <td>
                                <span class="bold">Reduces the target</span> to a <b>subhuman or non-human</b> status.<br>
                                <span class="yellow">💡 Hint:</span> comparisons to animals, objects, or entities, stripping away their humanity.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">Women are pigs.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Threatening</b></td>
                            <td>
                                <span class="bold">Explicit, direct (to target), threatening language or incitement to harm.</span> Expresses intent or encourages others to <b>take action against the target</b>.<br>
                                <span class="yellow">💡 Hint:</span> threats of physical, emotional, or privacy-related harm, direct threats or calls for harm, such as violence or violation.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">These women should be shot? Agreed!</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Derogation</b></td>
                            <td>
                                <span class="bold">Explicit derogatory, insulting, or demeaning</span> language, focusing on the target's <b>character, abilities, or physical attributes</b>.<br>
                                <span class="yellow">💡 Hint:</span> negative stereotypes, insults, or slurs.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">Women's football is so slow and clumsy.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Animosity</b></td>
                            <td>
                                <span class="bold">Implicit or subtle hostility,</span> often framed in a way that normalizes or downplays sexism via statements such as <b>backhanded compliments</b>.<br>
                                <span class="yellow">💡 Hint:</span> indirect insults or subtle forms of bias.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">Women are delicate flowers who need to be cherished.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Support of Hate Crimes</b></td>
                            <td>
                                <span class="bold">Glorifies, supports, or justifies hate crimes or discrimination.</span> Praise for violent actions, systemic discrimination, or organizations that perpetuate hate.<br>
                                <span class="yellow">💡 Hint:</span> expressions that endorse, justify, or deny hate-based violence or discrimination.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">Women shouldn't show that much skin; it's their own fault if they get raped.</span>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

            show_gbv_guideline2 = st.toggle("View Examples of Annotating GBV Text")
            if show_gbv_guideline2:
            # Show or hide the annotation guidelines based on user selection
                st.markdown("#### Examples of Annotating GBV Text")
                st.markdown(
                    """
                    <style>
                        table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        th, td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #f4f4f4;
                            font-weight: bold;
                        }
                        .bold { font-weight: bold; }
                        .italic { font-style: italic; }
                        .underline { text-decoration: underline; }
                        .yellow { color: #D4A017; } /* Golden yellow for tone label */
                    </style>

                    <table>
                        <colgroup>
                            <col style="width: 35%;">
                            <col style="width: 65%;">
                        </colgroup>
                        <tr>
                            <th>GBV Text</th>
                            <th>Annotation Example</th>
                        </tr>
                        <tr>
                            <td>Women should be shot? Agreed!</td>
                            <td>
                                <span class="bold underline">GBV Target:</span> women<br>
                                <span class="bold underline">GBV Form:</span> Threatening<br>
                            </td>
                        </tr>
                        <tr>
                            <td>I don't want to have to be around a breastfeeding woman, if she wants to do that she can do it at home.</td>
                            <td>
                                <span class="bold underline">GBV Target:</span> a breastfeeding woman<br>
                                <span class="bold underline">GBV Form:</span> Animosity<br>
                            </td>
                        </tr>
                        <tr>
                            <td>Women's football is so slow and clumsy.</td>
                            <td>
                                <span class="bold underline">GBV Target:</span> women<br>
                                <span class="bold underline">GBV Form:</span> Derogation<br>
                            </td>
                        </tr>
                        <tr>
                            <td>Women are pigs.</td>
                            <td>
                                <span class="bold underline">GBV Target:</span> women<br>
                                <span class="bold underline">GBV Form:</span> Dehumanisation<br>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
            
            
        # no more data available to assign
        if subset_size == 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.error("**NO DATA AVAILABLE!**  \nAll the examples have been assigned, and no more examples can be annotated.  \n\n Please **click [here]({NO_DATA_URL})** to return to Prolific and confirm your participation.  \n Please also **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
            st.markdown("If you meet any errors, feel free to message us. Thank you for your feedback and interest in our task -- we look forward to welcoming you in future rounds.")    
            st.stop()  # Stop further execution and display the message

        # if not pass the test -> no assignment to this user
        elif subset_size == 1 and not st.session_state.subset[0]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning("**ERROR!**  \nFailed test.")
            st.stop()
        
        # if subset is available, show annotation task
        elif 0 < subset_size <= TOTAL_CASES and st.session_state.subset[0]: # and subset_size > 0:
            
            ### Display a pair text for testing before starting annotation
            current_index = st.session_state.page
            num_completed_pairs = 0 
            
    
            ### Display current pair and start annotation
            if 0 <= current_index < subset_size: 
                pair_id = st.session_state.subset[current_index][0]
                pair = session.query(SourceDataset).get(pair_id)

                user_info1 = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
                if pair_id not in user_info1.pairs.split(", "):
                    existing_gbv_target = st.session_state.subset[current_index][1]
                    existing_gbv_form = st.session_state.subset[current_index][2]
                    existing_feedback = st.session_state.subset[current_index][3]
                else:
                    existing_gbv_target = ""
                    existing_gbv_form = ""
                    existing_feedback = ""

                # show progress bar for current example
                show_progress_bar(user_id, subset_size, session)
                st.success(f"**GBV EXAMPLE {current_index+1}**  \n\n {pair.gbv}")
                st.markdown("<br>", unsafe_allow_html=True)

                with st.container(border=True):
                    tab1, tab2 = st.tabs(["**SUBTASK 1 - GBV Target**", "**SUBTASK 2 - GBV Form**"])

                    with tab1:
                        pre_filled_target = existing_gbv_target if existing_gbv_target!="" else ""

                        # text highlighter
                        def find_indices(text, substring):
                            start = text.find(substring)
                            if start == -1:
                                return None  # Substring not found
                            end = start + len(substring)  # End index (exclusive)
                            return start, end
                        
                        st.markdown(
                            "<p style='font-size:14px; color:grey;'>" # font-style:italic;'>"
                            "Highlight the particular sub-category used to refer to the <b>woman/women or ideology being targeted</b> "
                            "(e.g. a breastfeeding woman, a girl, feminist, feminism, gender equality, etc.). "
                            "Select the text <b>from left to right</b> to apply the colour shown above, and <b>double-click</b> to remove the selection."
                            "</p>",
                            unsafe_allow_html=True
                        )
                        gbv_text = f"{pair.gbv}"
                        if pre_filled_target:
                            start, end = find_indices(gbv_text, pre_filled_target)
                            gbv_target_selected = text_highlighter(
                                text=gbv_text,
                                labels=[
                                    ("GBV Target", "lightgreen"),
                                ],
                                # Optionally specify pre-existing annotations:
                                annotations=[
                                    {"start": start, "end": end, "tag": "GBV Target"},
                                ],
                            )
                        else:
                            gbv_target_selected = text_highlighter(
                                text=gbv_text,
                                labels=[
                                    ("GBV target", "lightgreen"),
                                ],
                            )
                        if gbv_target_selected:
                            gbv_target_input = gbv_text[gbv_target_selected[0]["start"]:gbv_target_selected[0]["end"]]
                        else:
                            gbv_target_input = ""
                        
                        # Update gbv target in session_state subset -> convert tuple to list, update it, then assign back
                        temp_target_list = list(st.session_state.subset[current_index])  
                        temp_target_list[1] = gbv_target_input
                        st.session_state.subset[current_index] = tuple(temp_target_list) 

                    with tab2:
                        pre_selected_forms = existing_gbv_form.split(", ") if existing_gbv_form else []

                        gbv_forms1 = [""] + gbv_forms  # invalid value/empty is index=0
                        
                        # Allow up to 2 selections
                        selected_form = st.multiselect(
                            label=":grey[Select up to 2 forms for GBV text]",
                            options=gbv_forms1,
                            default=pre_selected_forms[:2],  # Pre-select existing labels, limited to 2
                            key=f"gbv_form_{user_id}_{pair.pair_id}_{current_index}",
                            max_selections=2,
                            placeholder=""
                        )

                        selected_form = ", ".join(selected_form)
                        temp_form_list = list(st.session_state.subset[current_index])  
                        temp_form_list[2] = selected_form 
                        st.session_state.subset[current_index] = tuple(temp_form_list)  
        
                    
                # Feedback
                st.markdown(f"##### **FEEDBACK**")
                feedback = [
                    "**TARGET CONFUSION:** Hard to select GBV targets for this example",
                    "**FORM CONFUSION:** Hard to select GBV forms for this example",
                    "**UNCOMFORTABLE:** I do not feel good when I see the example",
                    "**NONE**",
                ]
                
                # Pre-fill with existing label if available
                pre_selected_feedback = existing_feedback.split("@@@") if existing_feedback else []
                default_selected_feedback =  pre_selected_feedback[:-1] if pre_selected_feedback else []
                default_comment = pre_selected_feedback[-1] if pre_selected_feedback else ""

                # Checkbox feedback
                existence = [value in default_selected_feedback for value in feedback]
                
                feedback0 = st.checkbox(feedback[0], value=existence[0], key=f"feedback0_{user_id}_{pair.pair_id}_{current_index}")
                feedback1 = st.checkbox(feedback[1], value=existence[1], key=f"feedback1_{user_id}_{pair.pair_id}_{current_index}")
                feedback2 = st.checkbox(feedback[2], value=existence[2], key=f"feedback2_{user_id}_{pair.pair_id}_{current_index}")
                feedback3 = st.checkbox(feedback[3], value=existence[3], key=f"feedback3_{user_id}_{pair.pair_id}_{current_index}")
    
                feedback_list = [feedback0, feedback1, feedback2, feedback3]
                selected_feedback = []
                for j, feedback_cb in enumerate(feedback_list):
                    if feedback_cb:
                        selected_feedback.append(feedback[j])
                
                comment = st.text_area(
                            ":grey[Comment]",
                            value=default_comment,
                            key=f"comment_{user_id}_{pair.pair_id}_{current_index}",
                            placeholder="Add comments here for other feedback."
                        )
                selected_feedback.append(comment)
                selected_feedback = "@@@".join(selected_feedback)
                
                # Update feedback in session_state subset -> convert tuple to list, update it, then assign back
                temp_feedback_list = list(st.session_state.subset[current_index])  
                temp_feedback_list[3] = selected_feedback  
                st.session_state.subset[current_index] = tuple(temp_feedback_list)  
                st.markdown("<br>", unsafe_allow_html=True)
                
                ### Design action buttons per page
                col1, _, col3 = st.columns([1, 6, 1])
                with col1:
                    prev = st.button("Previous")     
                with col3:
                    save_and_next = st.button("Save and Next")   

                # Button 1 -- move to previous page
                if prev:
                    if current_index > 0:
                        st.session_state.page -= 1
                        st.rerun()
                    else:
                        st.warning("This is the first example.")
                           
                # Button 3 -- save results and move to next page
                if save_and_next:
                    labels = [gbv_target_input, selected_form, selected_feedback]

                    if "" not in labels:
                        # Record annotation per pair for this user and update pair completion number
                        update_annotation_record(user_id, pair_id, labels, session)

                        user_info2 = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
                        if str(pair_id) not in user_info2.pairs.split(", "): 
                            update_pair_completion(pair_id, session)
                            update_user_anno_info(user_id, pair_id, session)
                        st.session_state.page += 1
                        st.rerun()

                    elif "" in labels and current_index < subset_size: 
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning("⚠️ Please complete all subtasks before moving to the next page.")
                    
                    elif "" in labels and current_index == subset_size: 
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning("⚠️ Please complete all subtasks for each example.")


            elif current_index == subset_size: 
                user_action = session.query(SessionRecord).filter_by(user_id=user_id).first()

                ### Check if all pairs in the current session are completed
                num_completed_pairs = session.query(AnnotationRecord).filter(
                    AnnotationRecord.user_id == user_id,
                    AnnotationRecord.pair_id.in_([assignment.pair_id 
                                                for assignment in session.query(Assignment).filter_by(assign_id=user_action.assign_id)])
                ).filter(
                    or_(
                        not_(AnnotationRecord.gbv_target == ""),
                        not_(AnnotationRecord.gbv_form == ""),
                        not_(AnnotationRecord.feedback == ""),
                    )
                ).count()

                ### Update SessionRecord to record user status and elapsed time
                if subset_size == num_completed_pairs:
                    # Update user status to inactive (0) as all pairs in current assigned set are completed
                    user_action.status = 0  

                    # End the timer and calculate elapsed time in seconds/minutes
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) #/ 60
                    
                    # Update time in SessionRecord
                    user_action.time = elapsed_time
                    
                    session.commit()

                    # exit screen
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success(f"🎉 Thank you for completing this round of the task!")
                    st.markdown(f"To confirm your participation, please click [here]({COMPLETION_URL}) to return to Prolific and submit your completion.")
                    st.markdown("If the link does not work, please message us on Prolific.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.stop()  # Stop further execution and display the message
    
        else:
            st.warning("Something went wrong. Please contact us via Prolific for help.")
            st.stop()
        
        
main()
