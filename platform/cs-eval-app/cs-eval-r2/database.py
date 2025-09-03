import pandas as pd
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base, Session


# define the Base for all dataset tables
Base = declarative_base()

# Dataset 1: Source dataset
# store original hs-cs dataset 
class SourceDataset(Base):
    __tablename__ = "source_dataset"
    pair_id = Column(Integer, primary_key=True, autoincrement=True)
    gbv = Column(String, nullable=False)
    cs_human = Column(String, nullable=False)
    cs_1 = Column(String, nullable=False)
    cs_2 = Column(String, nullable=False)
    cs_3 = Column(String, nullable=False)
    type = Column(String, nullable=False)

    # dataset relationship
    completion = relationship("PairCompletion", back_populates="source")
    assign = relationship("Assignment", back_populates="source")


# Dataset 2: Assignment record per user
class Assignment(Base):
    __tablename__ = "assignment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('source_dataset.pair_id'), nullable=False)
    assign_id = Column(Integer, ForeignKey('session_record.assign_id'), nullable=False)

    # dataset relationship
    source = relationship("SourceDataset", back_populates="assign")
    session = relationship("SessionRecord", back_populates="assign")
    annotation_rec = relationship("AnnotationRecord", back_populates="assign")
  

# Dataset 3: Session action record 
# for user who return to task / incomplete task or who want to continue new subset of annotation
# for example, required 30 pairs per user per round, if the user done all -> active to inactive; if not -> keep active
# if active -> existing session has not been done -> assign same 30 pairs back to the user (record pair?)
# if inactive -> assign new 30 pairs to the user
class SessionRecord(Base):
    __tablename__ = "session_record"
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('annotator_info.user_id'), nullable=False)
    assign_id = Column(String, default='')
    status = Column(Integer, default=1)  # 1 - active, 0 - inactive
    time = Column(Integer, nullable=False)

    # dataset relationship
    assign = relationship("Assignment", back_populates="session")
    annotator = relationship("AnnotatorInfo", back_populates="session")
    

# Dataset 4: Annotation record
# store hs-cs pair, annotated label, and corresponding annotator
class AnnotationRecord(Base):
    __tablename__ = "annotation_record"
    anno_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('annotator_info.user_id'), nullable=False)
    pair_id = Column(Integer, ForeignKey('assignment.pair_id'), nullable=False)
    round_2 = Column(String, default='')

    # dataset relationship
    assign = relationship("Assignment", back_populates="annotation_rec")


# Dataset 5: Pair completion
# store number of completions per pair (e.g. limit 3 annotators per pair)
class PairCompletion(Base):
    __tablename__ = "pair_completion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('source_dataset.pair_id'), nullable=False)
    num_assign = Column(Integer, default=0)
    num_complete = Column(Integer, default=0)

    # dataset relationship
    source = relationship("SourceDataset", back_populates="completion")


# Dataset 6: Annotator information
# store annotator id from Prolific, metadata (e.g. demographic)
class AnnotatorInfo(Base):
    __tablename__ = "annotator_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    num_text = Column(Integer, default=0)
    passed = Column(Integer, default=0)  # 1 - passed, 0 - failed / not tested
    pairs = Column(String, default='') # record a list of completed pairs for each user -> avoid the case that annnotation of one pair is done and saved in num_complete, but the user refresh the page and click save button again to increase the num_complete

    # dataset relationship
    session = relationship("SessionRecord", back_populates="annotator")


def load_source_data(file_path: str, engine):
    """
    Load data from a tsv/csv file from a DataFrame into the BatchDataset.
    """
    df = pd.read_csv(file_path, sep='\t')  # tsv

    # check for required columns
    required_columns = {"pair_id", 
                        "gbv", 
                        "cs_human", 
                        "cs_1",
                        "cs_2",	
                        "cs_3",	
                        "type"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The file must contain these columns: {required_columns}")

    # create a session to insert rows into the database
    with Session(engine) as session:
        for _, row in df.iterrows():
            # Create BatchDataset instance
            new_row = SourceDataset(
                pair_id=row['pair_id'],
                gbv=row['gbv'],
                cs_human=row['cs_human'],
                cs_1=row['cs_1'],
                cs_2=row['cs_2'],
                cs_3=row['cs_3'],
                type=row['type'],
            )
            session.merge(new_row)  # merge to prevent duplicates
        session.commit()


# Initialise the database
def init_database(file_path: str, engine):
    Base.metadata.create_all(engine)
    
    try:
        load_source_data(file_path, engine)
    except Exception as e:
        print(f"Failed to load batch data: {e}")

