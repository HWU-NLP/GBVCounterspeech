import random
import os
from pathlib import Path
import pandas as pd
import numpy as np
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


TOTAL_CASES = 30  # number of pairs assigned per participant, defaul=30 per 30 mins
ITEMS_PER_PAGE = 1  # only display 1 case per page
NUM_ANNOTATOR_PER_ITEM = 1  # default=3 for maximum annotator per item, add 1 for tolerance

CUSTOMISED_USER_ID = "13aaabc"

# Prolific completion code - TODO: check code every time before deployment
COMPLETION_CODE = "C153LHOR"
COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C153LHOR"
STOP_BEFORE_TASK_CODE = "CI67DN9N"
STOP_BEFORE_TASK_URL = "https://app.prolific.com/submissions/complete?cc=CI67DN9N"
SCREEN_OUT_CODE = "CP6IEUPM"  
SCREEN_OUT_URL = "https://app.prolific.com/submissions/complete?cc=CP6IEUPM"
NO_DATA_CODE = "C1NH9W7U"
NO_DATA_URL = "https://app.prolific.com/submissions/complete?cc=C1NH9W7U"

# Create the SQLite engine and session
DATABASE_URL = "sqlite:///cs_anno.db"   # TODO: rename it based on annotation round e.g. pilot_db
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
    st.session_state.DEPLOY = False # DEPLOY mode for hosting on Prolific

# Initialise the database with dataset
init_database('data/cs_sample.tsv', engine)

# Instantiate a database session
session = SessionLocal()

# strategies
strategy_options = [
    "Empathy and Affiliation",
    "Warning of Consequence",
    "Hypocrisy or Contradiction",
    "Shaming or Labelling",
    "Denouncing",
    "Providing Facts",
    "Humour or Sarcasm",
    "Questioning"
]

# qualification questions for Section 1 - concept questions
concept_questions = [
    {"question": "Gender-based violence (GBV) includes only physical violence.", "answer": False},
    {"question": "Counterspeech is always directed at the perpetrator of GBV.", "answer": False},
    {"question": "A single counterspeech response can use multiple strategies.", "answer": True},
    {"question": "Providing factual information can be considered a counterspeech strategy.", "answer": True},
]

multiple_choice_questions = [
    {"question": "What is the expected output for this annotation task?",
     "options": ["Identify hate speech", "Label counterspeech strategies", "Write new counterspeech responses", "Label counterspeech targets"],
     "correct": ["Label counterspeech strategies"]},
    {"question": "Which of the following are examples of counterspeech strategies?",
     "options": ["Encouraging GBV", "Providing Facts", "Humour or Sarcasm", "Ignoring the hate speech"],
     "correct": ["Providing Facts", "Humour or Sarcasm"]},
    {"question": "If a counterspeech response explains the negative legal or social consequences of GBV, which strategy is being used? (Single choice questions)",
     "options": ["Empathy and Affiliation", "Warning of Consequence", "Humour or Sarcasm", "Questioning"],
     "correct": ["Warning of Consequence"]},
]

# qualification test - annotation practice samples
annotation_samples = [
    {"gbv": "Women can't handle money.",
    "cs": "What proof do you have? This stereotype comes from medieval myths, not real financial behavior.",
    "correct_strategies": "Providing Facts"},
    {"gbv": "We can't bring in millions of refugees, and we shouldn't just accept the ills of Islam.",
     "cs": "These are people in need—our brothers and sisters. We should help them.",
     "correct_strategies": "Empathy and Affiliation"},
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
    while len(subset) < TOTAL_CASES:
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

        existing_annotation = session.query(AnnotationRecord).filter_by(
            user_id=user_id, pair_id=pair.pair_id
        ).first()

        if not existing_annotation:
            subset.append((pair.pair_id, "", "")) 
            selected_pair_ids.append(pair.pair_id)     
        elif any(
            getattr(existing_annotation, col) == ""
            for col in ["strategy", "feedback"]
        ):
            subset.append((pair.pair_id, "", ""))
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
            session.add(AnnotationRecord(
                user_id=user_id, 
                pair_id=pair_id, 
                strategy="",
                feedback = "", 
            ))
        else:
            # Update the label if needed (optional)
            existing_annotation.strategy = ""
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
        annotation.strategy = labels[0]
        annotation.feedback = labels[1]
    else:
        session.add(AnnotationRecord(
            user_id=user_id, 
            pair_id=pair_id, 
            strategy=labels[0],
            feedback = labels[1],
            ))
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
    record = SessionRecord(user_id=user_id, assign_id=assign_id, status=1, time=0) #, assign_n_set=str(len(subset)))

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

    subset = assign_new_subset(user_id, assign_id, session)

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
                        annotation.strategy, 
                        annotation.feedback, 
                        )) 

        else:
            # If annotation does not exist, leave it empty
            subset.append((assignment.pair_id, "", ""))  

    return subset, record

def get_max_button_size(strategy):
    """
    Calculate the max width and height for buttons
    """
    max_width = max([len(strategy) for strategy in strategy]) * 10  # Estimate width
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
            not_(AnnotationRecord.strategy == ""),
            not_(AnnotationRecord.feedback == ""),
        )
    ).count()

    # Calculate progress for progress bar
    progress = num_completed_pairs / subset_size 
    
    # display progress bar and progress text
    st.markdown("<br>", unsafe_allow_html=True)
    progress_text = f"⏳ Annotation Progress: {num_completed_pairs}/{subset_size}"
    st.progress(progress, text=progress_text)

def main():
    ### Custom CSS to ensure all buttons have the same height
    max_width, max_height = get_max_button_size(strategy_options)

    st.markdown(f"""
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
    """, unsafe_allow_html=True)


    ### Load Prolific ID and assign pairs to users based on 3 scenarios
    start_time = time.time()  # Start the timer
    user_id = get_user_id()
    
    ### Qualification test for new annotator
    # -> if page == -1 && no passed record of test in AnnotatorInfo
    # -> else load from the first incompleted pair
    
    if st.session_state.page == -1:  
        # Create sidebar with exit button for participants who stopped before formal task
        with st.sidebar:
            st.warning("**⏹️ EXIT BEFORE FORMAL ANNOTATION**  \nIf you have partially completed the task but wish to exit, click the button below to confirm your participation and exit.")
            exit1 = st.button("Exit GBV Counterspeech Annotation ")    
            if exit1:
                st.session_state.exit_clicked1 = True
                
                st.success(f"Thank you for participating in the task! Please click **[here]({STOP_BEFORE_TASK_URL})** to confirm your participation on Prolific.  \n\n Please **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("You may now close this window!")
                st.stop()
        
        user_info = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
        
        if not user_info or not user_info.passed:
            ### Task title
            st.title("Annotation Qualification Test")
        
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
                                       key=f"{user_id}_TF_{idx+1}") 
                
                if response_TF is not None:  # Ensure user selects an answer
                    if (response_TF == "True" and q["answer"]) or (response_TF == "False" and not q["answer"]):
                        s1_score += 1
            
            st.markdown("<br>", unsafe_allow_html=True)

            ## Multiple-Choice Questions
            st.markdown("##### 🟢 (II) Multiple-Choice Questions")
            
            # Add partial score to total score for multiple choice questions
            for idx, q in enumerate(multiple_choice_questions):
                response_MCQ = st.multiselect(label=f"**{idx+5}. {q['question']}**", 
                                              options=q["options"], 
                                              key=f"{user_id}_MCQ_{idx+5}")
                
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
            s2_total = len(annotation_samples) 
            
            for idx, sample in enumerate(annotation_samples):
                with st.container(border=True):
                    st.success(f"**TEST PAIR EXAMPLE {idx + 1}**  \n\n **GBV:** {sample['gbv']}  \n\n **CS:** {sample['cs']}")
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(f"###### **CS Strategy**")
                    selected_test_strategies = st.multiselect(label=":grey[Eight different strategies identified in CS. Provided in the sidebar are the **definitions, tone** you should look for and **examples** for each strategy. You may select **up to 3 strategies** for counterspeech.]",
                                                options=strategy_options,
                                                key=f"strategy_{idx}_{user_id}")

                    if sample["correct_strategies"] in selected_test_strategies and len(selected_test_strategies) <= 3:
                        s2_score += 1
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
            
            if not st.session_state.submit_and_correct and not st.session_state.submit_but_wrong and not st.session_state.start_task and not start:
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
            
            if st.session_state.submit_but_wrong and not start:
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Your total score: {accuracy:.1f}**")
                st.error("❌ Unfortunately, you did not pass.  \n Please review the guidelines and examples carefully, and try to submit your answers again.")
                st.markdown("If you do not want to continue the task, please click the **Exit GBV Counterspeech Annotation** button in the sidebar to confirm your participation.")
              
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

    
    # subset = []
    if st.session_state.start_task and st.session_state.page >= 0:

        record = get_user_action(user_id, session)
        if not record:
            subset, record = handle_scenario_1(user_id, session)
        elif record.status == 0:
            subset, record = handle_scenario_2(user_id, record, session)
        elif record.status == 1:
            subset, record = handle_scenario_3(user_id, record, session)
        else:
            st.error("No data has been assigned.") 

        st.session_state.subset = subset
        st.session_state.start_task = False
        update_user_test_info(user_id, session)
        st.rerun()
        
    #================================================================================================
    ### Start annotation
    
    subset_size = len(st.session_state.subset)
            
    if st.session_state.page >= 0:
        ### Create sidebar to present annotation guideline
        # # view/hide annotation guidelines toggle sync
        with st.sidebar:
            st.warning("**⏹️ EXIT BEFORE COMPLETION**  \nIf you have partially completed the annotations but wish to exit the task, click the button below to confirm your participation and exit.")
            exit2 = st.button("Exit GBV Counterspeech Annotation")    
            if exit2:
                st.session_state.exit_clicked2 = True
                st.success(f"Thank you for participanting in the task! Please click **[here]({SCREEN_OUT_URL})** to confirm your participation on Prolific.  \n Please **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("You may now close this window!")
                st.stop()

            st.markdown(f"------")
            st.subheader("🟢 Task Overview of GBV Counterspeech Annotation")
            st.markdown("You will see pairs of GBV text, and the counterspeech written in response to it. For each of these pairs, you need to look at the counterspeech and **assign labels**:")
            st.markdown("* **CS Strategy:** You need to label what kind of strategy was used to counter GBV text. For example, is the response humorous and sarcastic? There are eight options: Empathy and Affiliation, Warning of Consequence, Hypocrisy or Contradiction, Shaming or Labelling, Denouncing, Providing Facts, Humour or Sarcasm, and Questioning. You may select **up to 3 strategies** if needed.")
            st.markdown("**Feedback** is also welcome after annotating each pair to see if any issues you meet during counterspeech annotation process.")

            st.markdown(f"------")
            st.subheader("🟢 GBV Counterspeech Annotation Guidelines")
            st.markdown("<br>", unsafe_allow_html=True)
            
            show_cs_guideline1 = st.toggle("View CS Strategy Annotation Guideline")
            if show_cs_guideline1:
            # # Show or hide the annotation guidelines based on user selection
                st.markdown("#### CS Strategy Annotation Guideline")
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
                            <col style="width: 25%;">
                            <col style="width: 75%;">
                        </colgroup>
                        <tr>
                            <th>Strategy</th>
                            <th>Definition</th>
                        </tr>
                        <tr>
                            <td><b>Empathy and Affiliation</b></td>
                            <td>
                                <span class="bold">Focuses on promoting understanding</span>, fostering peace and finding common ground.<br>
                                <span class="yellow">📣 Tone:</span> Kind, compassionate, understanding language.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">These are people in need—our brothers and sisters. We should help them.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Warning of Consequence</b></td>
                            <td>
                                <span class="bold">Cautioning the speaker</span> about the impact of their words via potential negative outcomes, such as legal, social, or personal consequences.<br>
                                <span class="yellow">📣 Tone:</span> Serious, cautionary or urgent.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">Ignoring history risks hurting future generations by not teaching them the lessons of the past.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Hypocrisy or Contradiction</b></td>
                            <td>
                                <span class="bold">Discredit the argument</span> through critical analysis by pointing out inconsistencies, illogical reasoning, contradictions, or double standards.<br>
                                <span class="yellow">📣 Tone:</span> Critical, logical, analytical.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">
                                <b>GBV:</b> Zionist and Jewish control of America is the main cause of global terrorism and conflict.<br>
                                <b>CS:</b> Isn't colonialism and Western intervention abroad what led to much of the extremism?
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Shaming or Labelling</b></td>
                            <td>
                                <span class="bold">Direct and confrontational, including personal attacks</span>. Attacks the speaker by using negative labels or terms to emphasise the inappropriateness of their behaviour.<br>
                                <span class="yellow">📣 Tone:</span> Accusatory, confrontational, disapproving.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic">It seems you've become intolerant to homosexuals, maybe reflect on why love bothers you so much.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Denouncing</b></td>
                            <td>
                                <span class="bold">Refuting without personal attacks</span>. Condemns or rejects the hateful views, stating that they are wrong, unacceptable, harmful, etc.<br>
                                <span class="yellow">📣 Tone:</span> Firm, direct, authoritative.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic"><b>GBV:</b> I miss the good old days when we had black domestic workers.<br>
                                <b>CS:</b> The West got rid of slavery. But we've never really left racism.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Providing Facts</b></td>
                            <td>
                                <span class="bold">Correct misinformation and false narratives</span>. Uses facts, statistics, or evidence to counter false claims.<br>
                                <span class="yellow">📣 Tone:</span> Factual, corrective, neutral.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic"><b>GBV:</b> Women can't handle money.<br>
                                <b>CS:</b> What proof do you have? This stereotype comes from medieval myths, not real financial behavior.</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Humour or Sarcasm</b></td>
                            <td>
                                <span class="bold">Lighthearted, diminishes the comment by ridiculing it</span> (or the perpetrator) via humour, irony, or sarcasm.<br>
                                <span class="yellow">📣 Tone:</span> Playful, humorous, mocking.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic"><b>GBV:</b> Jews are why the World Cup is so expensive.<br>
                                <b>CS:</b> Oh, I thought it was players' salaries... but sure, let's blame the usual Jewish conspiracy!</span>
                            </td>
                        </tr>
                        <tr>
                            <td><b>Questioning</b></td>
                            <td>
                                <span class="bold">Exposes flaws in the argument</span> by probing its assumptions, logic, or intentions, often through asking pointed or clarifying questions. <br>
                                <span class="yellow">📣 Tone:</span> Inquisitive, probing, reflective.<br>
                                <span class="bold italic underline">Example</span><br>
                                <span class="italic"><b>GBV:</b> All Muslims are incompatible with any Western society.<br>
                                <b>CS:</b> I'm not sure what you are trying to say by incompatible. What do you mean by incompatible?</span>
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
        elif 0 < subset_size <= TOTAL_CASES and st.session_state.subset[0]: 
            
            ### Display a pair text for testing before starting annotation
            current_index = st.session_state.page
            num_completed_pairs = 0 
            
            ### Display current pair and start annotation
            if 0 <= current_index < subset_size: 
                pair_id = st.session_state.subset[current_index][0]
                pair = session.query(SourceDataset).get(pair_id)

                user_info1 = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
                if pair_id not in user_info1.pairs.split(", "):
                    existing_strategy = st.session_state.subset[current_index][1]
                    existing_feedback = st.session_state.subset[current_index][2]
                else:
                    existing_strategy = ""
                    existing_feedback = ""

                # show progress bar for current example
                show_progress_bar(user_id, subset_size, session)
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"**PAIR EXAMPLE {current_index+1}**  \n\n **GBV:** {pair.gbv}  \n\n**CS:** {pair.counterspeech}")
                st.markdown("<br>", unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown(f"##### **CS Strategy**")
                    
                    # Pre-fill with existing label if available
                    pre_selected_strategies = existing_strategy.split(", ") if existing_strategy else []
                    strategies1 = [""] + strategy_options  

                    # Allow up to 2/3 selections
                    selected_strategies = st.multiselect(
                        label=":grey[Eight different strategies identified in CS. Provided in the sidebar are the **definitions, tone** you should look for and **examples** for each strategy. You may select **up to 3 strategies** for counterspeech.]",
                        options=strategies1,
                        default=pre_selected_strategies[:3],  # Pre-select existing labels, limited to 2
                        key=f"strategy_{user_id}_{pair_id}_{current_index}",
                        max_selections=3,
                        placeholder=""
                    )

                    selected_strategies = ", ".join(selected_strategies)
                    
                    # Update strategy in session_state subset -> convert tuple to list, update it, then assign back
                    temp_strategy_list = list(st.session_state.subset[current_index])  
                    temp_strategy_list[1] = selected_strategies  
                    st.session_state.subset[current_index] = tuple(temp_strategy_list)  

                    st.markdown("<br>", unsafe_allow_html=True)

                # Feedback
                st.markdown(f"##### **FEEDBACK**")
                feedback = [
                    "**STRATEGY CONFUSION:** Hard to choose the correct strategies for this example",
                    "**MISMATCH:** CS refers to a completely different subject. E.g. HS related to **race**, CS related to **feminism**",
                    "**PARTIAL MATCH / INDIRECT:** CS sort of addresses the issue, but not in a straightforward way. E.g. HS is against **women**, but CS uses words like **feminism**",
                    "**NOT PERSUASIVE:** CS addresses the issue, but I didn't find it very convincing",
                    "**OTHER:** CS is not good for other reasons, such as being uninformative, vague, ambiguous. E.g. an uninformative CS response such as **'Why do you think that way?'** without any further text",
                    "**NONE**",
                ]
                feedback_display = [
                    "**STRATEGY CONFUSION:** Hard to choose the correct strategies for this example",
                    "**MISMATCH:** CS refers to a completely different subject.  \n :green[_E.g. HS related to **race**, CS related to **feminism**_]",
                    "**PARTIAL MATCH / INDIRECT:** CS sort of addresses the issue, but not in a straightforward way.  \n :green[_E.g. HS is against **women**, but CS uses words like **feminism**_]",
                    "**NOT PERSUASIVE:** CS addresses the issue, but I didn't find it very convincing",
                    "**OTHER:** CS is not good for other reasons, such as being uninformative, vague, ambiguous.  \n :green[_E.g. an uninformative CS response such as **'Why do you think that way?'** without any further text_]",
                    "**NONE**",
                ]
                
                # Pre-fill with existing label if available
                pre_selected_feedback = existing_feedback.split("@@@") if existing_feedback else []
                default_selected_feedback =  pre_selected_feedback[:-1] if pre_selected_feedback else []
                default_comment = pre_selected_feedback[-1] if pre_selected_feedback else ""

                # Checkbox feedback
                existence = [feedback_display[i] if value in default_selected_feedback else "" for i, value in enumerate(feedback)]
                
                feedback0 = st.checkbox(feedback_display[0], value=existence[0], key=f"feedback0_{user_id}_{pair_id}_{current_index}")
                feedback1 = st.checkbox(feedback_display[1], value=existence[1], key=f"feedback1_{user_id}_{pair_id}_{current_index}")
                feedback2 = st.checkbox(feedback_display[2], value=existence[2], key=f"feedback2_{user_id}_{pair_id}_{current_index}")
                feedback3 = st.checkbox(feedback_display[3], value=existence[3], key=f"feedback3_{user_id}_{pair_id}_{current_index}")
                feedback4 = st.checkbox(feedback_display[4], value=existence[4], key=f"feedback4_{user_id}_{pair_id}_{current_index}")
                feedback5 = st.checkbox(feedback_display[5], value=existence[5], key=f"feedback5_{user_id}_{pair_id}_{current_index}")

                feedback_list = [feedback0, feedback1, feedback2, feedback3, feedback4, feedback5]
                selected_feedback = []
                for j, feedback_cb in enumerate(feedback_list):
                    if feedback_cb:
                        selected_feedback.append(feedback[j])
                
                comment = st.text_area(
                            ":grey[Comment]",
                            value=default_comment,
                            key=f"comment_{user_id}_{pair_id}_{current_index}",
                            placeholder="Add comments here for other feedback."
                        )
                selected_feedback.append(comment)
                selected_feedback = "@@@".join(selected_feedback)

                # Update feedback in session_state subset -> convert tuple to list, update it, then assign back
                temp_feedback_list = list(st.session_state.subset[current_index])  
                temp_feedback_list[2] = selected_feedback  
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
                    labels = [selected_strategies, selected_feedback]

                    if "" not in labels:
                        # record completed annotation per pair (including the case - user changed annotations and save again)
                        update_annotation_record(user_id, pair_id, labels, session)
                        
                        user_info2 = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
                        if str(pair_id) not in user_info2.pairs.split(", "):
                            update_pair_completion(pair_id, session)
                            update_user_anno_info(user_id, pair_id, session)
                        st.session_state.page += 1
                        st.rerun()

                    elif "" in labels and current_index < subset_size:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning(f"⚠️ Please complete all subtasks before moving to the next page.")
                    
                    elif "" in labels and current_index == subset_size: 
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning(f"⚠️ Please complete all subtasks for each example.")

            elif current_index == subset_size: 
                user_action = session.query(SessionRecord).filter_by(user_id=user_id).first()

                ### Check if all pairs in the current session are completed
                num_completed_pairs = session.query(AnnotationRecord).filter(
                    AnnotationRecord.user_id == user_id,
                    AnnotationRecord.pair_id.in_([assignment.pair_id 
                                                for assignment in session.query(Assignment).filter_by(assign_id=user_action.assign_id)])
                ).filter(
                    or_(
                        not_(AnnotationRecord.strategy == ""),
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

        else:
            st.warning("Something went wrong. Please contact us via Prolific for help.")
            st.stop()

main()

