import random
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import re

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
NUM_ANNOTATOR_PER_ITEM = 3  # default=3 for maximum annotator per item, add 1 for tolerance

CUSTOMISED_USER_ID = "42bdsc"

# Prolific completion code - TODO: check code every time before deployment
COMPLETION_CODE = "C16GKLGT"
COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C16GKLGT"
SCREEN_OUT_CODE = "C8M53ZPV"  
SCREEN_OUT_URL = "https://app.prolific.com/submissions/complete?cc=C8M53ZPV"
NO_DATA_CODE = "C1PZ8OX8"
NO_DATA_URL = "https://app.prolific.com/submissions/complete?cc=C1PZ8OX8"

# Create the SQLite engine and session
DATABASE_URL = "sqlite:///cs_eval_r2.db"  # TODO: rename it based on annotation round e.g. pilot_db
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)


# Initialise session state for guidelines toggle, page setting
if 'page' not in st.session_state:
    st.session_state.page = 0  # -1 for test pair, 0 for formal pairs
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
    st.session_state.start_task = True #False
if 'page_clicked' not in st.session_state:
    st.session_state.page_clicked = False
if 'shuffled' not in st.session_state:
    st.session_state.shuffled = False  # for shuffling the cs responses
if 'shuffled_list' not in st.session_state:
    st.session_state.shuffled_list = []  
if 'DEPLOY' not in st.session_state:
    st.session_state.DEPLOY = False # DEPLOY mode for hosting on Prolific

# Initialise the database with dataset
init_database('data/eval_sample_r2.tsv', engine)

# Instantiate a database session
session = SessionLocal()

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
            subset.append((pair.pair_id, "", )) 
            selected_pair_ids.append(pair.pair_id)     
        elif any(
            getattr(existing_annotation, col) == ""
            for col in ["round_2"]
        ):
            subset.append((pair.pair_id, ""))
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
                round_2="",
            ))
        else:
            # Update the label if needed (optional)
            existing_annotation.round_2 = ""
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
        annotation.round_2 = labels[0]
    else:
        session.add(AnnotationRecord(
            user_id=user_id, 
            pair_id=pair_id, 
            round_2=labels[0],
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
                        annotation.round_2, 
                        )) 

        else:
            # If annotation does not exist, leave it empty
            subset.append((assignment.pair_id, ""))  

    return subset, record


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
            not_(AnnotationRecord.round_2 == ""),
        )
    ).count()

    # Calculate progress for progress bar
    progress = num_completed_pairs / subset_size 
    
    # display progress bar and progress text
    st.markdown("<br>", unsafe_allow_html=True)
    progress_text = f"⏳ Annotation Progress: {num_completed_pairs}/{subset_size}"
    st.progress(progress, text=progress_text)

def check_selection(selected_list):
    """
    Check if the user has selected exactly 3 counterspeech responses.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    
    for item in selected_list:
        if 'None' in item:
            st.warning("⚠️ Please give feedback for all counterspeech texts before submitting.")
            return False
    else:
        st.success("✅ Selection complete! You can now submit and proceed to the next example.")
        return True
    
def clean_dict(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith('_')}

def main():
    ### Custom CSS to ensure all buttons have the same height
    max_width, max_height = 1000, 60

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
            st.warning("**⏹️ EXIT BEFORE COMPLETION**  \nIf you have partially completed the task but wish to exit, click the button below to confirm your participation and exit.")
            exit2 = st.button("Exit GBV Counterspeech Evaluation")    
            if exit2:
                st.session_state.exit_clicked2 = True
                st.success(f"Thank you for participating in the task! Please click **[here]({SCREEN_OUT_URL})** to confirm your participation on Prolific.  \n Please **return the study** on Prolific and we will still compensate you with a bonus based on the your effort.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("You may now close this window!")
                st.stop()

            st.markdown(f"------")
            st.subheader("🟢 Task Overview")
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
                    existing_round2 = st.session_state.subset[current_index][1]
                else:
                    existing_round2 = ""

                # show progress bar for current example
                show_progress_bar(user_id, subset_size, session)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"**GBV EXAMPLE {current_index+1}**  \n\n {pair.gbv}")
                st.markdown("<br>", unsafe_allow_html=True)
                
                cs_r2 = [
                    ("cs_1", pair.cs_1),
                    ("cs_2", pair.cs_2),
                    ("cs_3", pair.cs_3),
                    ("cs_human", pair.cs_human)
                ]
                
                if not st.session_state.shuffled:
                    random.seed(st.session_state.page)
                    st.session_state.shuffled_list = random.sample(cs_r2, k=len(cs_r2))
                    st.session_state.shuffled = True  # Set the flag to indicate shuffling has been done
                cs_r2_shuffled = st.session_state.shuffled_list

                with st.container(border=True):
                    st.markdown(f"**Counterspeech 1:**  \n  {cs_r2_shuffled[0][1]}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1_0_0, col2_0_0= st.columns([8,1])
                    with col1_0_0:
                        st.markdown("**Does the response directly and appropriately address the harmful content?**")
                    with col2_0_0:
                        cs_r2_0_0 = st.feedback(options="thumbs", key=f"cs_r2_0_0_{user_id}_{pair_id}_{current_index}")
                    
                    col1_0_1, col2_0_1= st.columns([8,1])
                    with col1_0_1:
                        st.markdown("**Does the response feel persuasive or effective?**")
                    with col2_0_1:
                        cs_r2_0_1 = st.feedback(options="thumbs", key=f"cs_r2_0_1_{user_id}_{pair_id}_{current_index}")

                    col1_0_2, col2_0_2= st.columns([8,1])
                    with col1_0_2:
                        st.markdown("**Do you think the response could promote positive and educational dialogue?**")
                    with col2_0_2:
                        cs_r2_0_2 = st.feedback(options="thumbs", key=f"cs_r2_0_2_{user_id}_{pair_id}_{current_index}")
                        
                with st.container(border=True):
                    st.markdown(f"**Counterspeech 2:**  \n  {cs_r2_shuffled[1][1]}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1_1_0, col2_1_0= st.columns([8,1])
                    with col1_1_0:
                        st.markdown("**Does the response directly and appropriately address the harmful content?**")
                    with col2_1_0:
                        cs_r2_1_0 = st.feedback(options="thumbs", key=f"cs_r2_1_0_{user_id}_{pair_id}_{current_index}")
                    
                    col1_1_1, col2_1_1= st.columns([8,1])
                    with col1_1_1:
                        st.markdown("**Does the response feel persuasive or effective?**")
                    with col2_1_1:
                        cs_r2_1_1 = st.feedback(options="thumbs", key=f"cs_r2_1_1_{user_id}_{pair_id}_{current_index}")
                    
                    col1_1_2, col2_1_2= st.columns([8,1])
                    with col1_1_2:
                        st.markdown("**Do you think the response could promote positive and educational dialogue?**")
                    with col2_1_2:
                        cs_r2_1_2 = st.feedback(options="thumbs", key=f"cs_r2_1_2_{user_id}_{pair_id}_{current_index}")
                
                with st.container(border=True):
                    st.markdown(f"**Counterspeech 3:**  \n  {cs_r2_shuffled[2][1]}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1_2_0, col2_2_0= st.columns([8,1])
                    with col1_2_0:
                        st.markdown("**Does the response directly and appropriately address the harmful content?**")
                    with col2_2_0:
                        cs_r2_2_0 = st.feedback(options="thumbs", key=f"cs_r2_2_0_{user_id}_{pair_id}_{current_index}")
                    
                    col1_2_1, col2_2_1= st.columns([8,1])
                    with col1_2_1:
                        st.markdown("**Does the response feel persuasive or effective?**")
                    with col2_2_1:
                        cs_r2_2_1 = st.feedback(options="thumbs", key=f"cs_r2_2_1_{user_id}_{pair_id}_{current_index}")
                    
                    col1_2_2, col2_2_2= st.columns([8,1])
                    with col1_2_2:
                        st.markdown("**Do you think the response could promote positive and educational dialogue?**")
                    with col2_2_2:
                        cs_r2_2_2 = st.feedback(options="thumbs", key=f"cs_r2_2_2_{user_id}_{pair_id}_{current_index}")
                
                with st.container(border=True):
                    st.markdown(f"**Counterspeech 4:**  \n  {cs_r2_shuffled[3][1]}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1_3_0, col2_3_0= st.columns([8,1])
                    with col1_3_0:
                        st.markdown("**Does the response directly and appropriately address the harmful content?**")
                    with col2_3_0:
                        cs_r2_3_0 = st.feedback(options="thumbs", key=f"cs_r2_3_0_{user_id}_{pair_id}_{current_index}")
                    
                    col1_3_1, col2_3_1= st.columns([8,1])
                    with col1_3_1:
                        st.markdown("**Does the response feel persuasive or effective?**")
                    with col2_3_1:
                        cs_r2_3_1 = st.feedback(options="thumbs", key=f"cs_r2_3_1_{user_id}_{pair_id}_{current_index}")

                    col1_3_2, col2_3_2= st.columns([8,1])
                    with col1_3_2:
                        st.markdown("**Do you think the response could promote positive and educational dialogue?**")
                    with col2_3_2:
                        cs_r2_3_2 = st.feedback(options="thumbs", key=f"cs_r2_3_2_{user_id}_{pair_id}_{current_index}")

                cs_r2_list = [
                    (str(cs_r2_shuffled[0][0]), str(cs_r2_0_0), str(cs_r2_0_1), str(cs_r2_0_2)), 
                    (str(cs_r2_shuffled[1][0]), str(cs_r2_1_0), str(cs_r2_1_1), str(cs_r2_1_2)), 
                    (str(cs_r2_shuffled[2][0]), str(cs_r2_2_0), str(cs_r2_2_1), str(cs_r2_2_2)), 
                    (str(cs_r2_shuffled[3][0]), str(cs_r2_3_0), str(cs_r2_3_1), str(cs_r2_3_2)), 
                ]
                    
                is_correctly_selected = check_selection(cs_r2_list)
                
                selected_cs_r2_list = cs_r2_list.copy()
                selected_cs_r2_list = [str(item) for item in selected_cs_r2_list] 
                selected_cs_r2_text = "@@@".join(selected_cs_r2_list)  # Convert to string for storage

                # Update cs_r2 in session_state subset -> convert tuple to list, update it, then assign back
                temp_cs_r2_list = list(st.session_state.subset[current_index])  
                temp_cs_r2_list[1] = selected_cs_r2_text  
                st.session_state.subset[current_index] = tuple(temp_cs_r2_list)  
                st.markdown("<br>", unsafe_allow_html=True)

        
                ### Design action buttons per page
                col1, _, col3 = st.columns([1, 6, 1])     
                with col3:
                    save_and_next = st.button("Save and Next", disabled=not is_correctly_selected)  # Disable if not correctly selected

                # Button -- save results and move to next page
                if save_and_next:
                    labels = [selected_cs_r2_text] 
                
                    # record completed annotation per pair (including the case - user changed annotations and save again)
                    update_annotation_record(user_id, pair_id, labels, session)
                    
                    user_info2 = session.query(AnnotatorInfo).filter_by(user_id=user_id).first()
                    if str(pair_id) not in user_info2.pairs.split(", "):
                        update_pair_completion(pair_id, session)
                        update_user_anno_info(user_id, pair_id, session)
                    st.session_state.shuffled = False
                    st.session_state.page += 1
                    st.rerun()

            elif current_index == subset_size: 
                user_action = session.query(SessionRecord).filter_by(user_id=user_id).first()

                ### Check if all pairs in the current session are completed
                num_completed_pairs = session.query(AnnotationRecord).filter(
                    AnnotationRecord.user_id == user_id,
                    AnnotationRecord.pair_id.in_([assignment.pair_id 
                                                for assignment in session.query(Assignment).filter_by(assign_id=user_action.assign_id)])
                ).filter(
                    or_(
                        not_(AnnotationRecord.round_2 == ""),
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

