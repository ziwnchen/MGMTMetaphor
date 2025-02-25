import json
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import hashlib

# Utility functions for exporting processed datasets
# applied to both trend and agent matching process

## 1. generate sentence id
def generate_sentence_id_explicit(row, doc_id, sentence, word_loc): # for df that contains explicit measures
    unique_str = f"{row[doc_id]}_{row[sentence]}_{row[word_loc]}"
    return hashlib.md5(unique_str.encode()).hexdigest()

def generate_sentence_id_implicit(row, doc_id, object_mask): # for df that contains implicit measures
    unique_str = f"{row[doc_id]}_{row[object_mask]}"
    return hashlib.md5(unique_str.encode()).hexdigest()

def generate_sentence_id_idx(row, doc_id, idx): # use idx to represent sentence (after deduplication, sentence at each row is unique)
    unique_str = f"{row[doc_id]}_{row[idx]}"
    return hashlib.md5(unique_str.encode()).hexdigest()

def explicit_sent_hash(df):
    df['sentence_id'] = df.progress_apply(lambda x: generate_sentence_id_explicit(x, 'doc_id', 'mgmt_sents','focal_char_span'), axis=1) # hash the sentence id
    return df

def implicit_sent_hash(df):
    df['sentence_id'] = df.progress_apply(lambda x: generate_sentence_id_implicit(x, 'doc_id', 'object_mask'), axis=1) # hash the sentence id
    return df

def idx_sent_hash(df):
    df['idx'] = df.index
    df['sentence_id'] = df.progress_apply(lambda x: generate_sentence_id_implicit(x, 'doc_id', 'idx'), axis=1) # hash the sentence id
    return df

# generate age
def generate_age(year, birth_year):
    if pd.isnull(birth_year):
        return np.nan
    elif birth_year == 'Missing':
        return np.nan
    else:
        return int(year) - int(birth_year)

def calculate_decade(year):
    return int(year/10)*10

## 2. functions for explicit measures
def remove_intransitive(df):
    df = df[df['if_intransitive']==False]
    df = df[(df['WSD_pred']>0) & (df['WSD_conf']>0.9)]
    df.reset_index(drop=True,inplace=True)
    return df

def gen_true_label(label, confidence):
    if confidence>0.9: # threshold for confidence
        return int(label)
    else:
        return 999 

# def create_pb_labels(df, reverse_secondary_map):
#     df['pb_primary_label']= df.apply(lambda x: gen_true_label(x['pb_primary_predictions'], x['pb_primary_confidences']), axis=1)
#     df['pb_secondary_label']= df.apply(lambda x: gen_true_label(x['pb_subcategory_predictions'], x['pb_subcategory_confidences']), axis=1)
#     # given the classification, create a dummy variable for the primary category, and then run logit model
#     df['IsPerson'] = df.apply(lambda row: 1 if (row['pb_primary_label']==0) and (row['pb_secondary_label'] not in [0,4,999]) else 0, axis=1) # 0: others, 4: household, 999: low confidence or not classified
#     df['secondaryLabel'] = df['pb_secondary_label'].map(reverse_secondary_map)
#     return df

def export_explicit_df(df, path):
    selected_columns = ['year','dataset','doc_id','sentence_id', 'IsPerson', 'secondaryLabel']
    df[selected_columns].to_csv(path, index=False)
    return

def process_explicit_data(data_path, filename, processing_steps, export_filename):
    # Load dataset
    df = pd.read_csv(data_path + filename)
    print(f"Dataset before processing: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Apply all processing steps in sequence
    for step in processing_steps:
        df = step(df)

    # Export processed dataset
    print(f"Dataset after processing: {df.shape[0]} rows, {df.shape[1]} columns.")
    export_explicit_df(df, data_path + export_filename)
    return

def export_explicit_agent_match(df, path):
    selected_columns = ['year','doc_id','sentence_id','IsPerson','secondaryLabel','wikidata_id','party','birth_year','age','gender',
    'if_college_ed','if_business_occupation','if_business_ed','dataset','decade']
    df[selected_columns].to_csv(path, index=False)
    return

## 3. functions for implicit measures
def export_implicit_df(df, path):
    selected_columns = ['year','dataset','doc_id','sentence_id','object', 'subgroup_orig_syn_ratio', 'top_subgroup']
    df[selected_columns].to_csv(path, index=False)
    return

def process_implicit_data(data_path, filename, processing_steps, export_filename):
    # Load dataset
    df = pd.read_pickle(data_path + filename)
    df.reset_index(drop=True,inplace=True)
    
    # Apply all processing steps in sequence
    for step in processing_steps:
        df = step(df)

    # Export processed dataset
    export_implicit_df(df, data_path + export_filename)
    return

def export_implicit_agent_match(df, path):
    selected_columns = ['year','doc_id','sentence_id','object', 'top_subgroup','subgroup_orig_syn_ratio','wikidata_id',
    'party','birth_year','age','gender','if_college_ed','if_business_occupation','if_business_ed','dataset','decade','object_category','if_relational_subjectivity']
    df[selected_columns].to_csv(path, index=False)
    return

def gen_group_category(df, category_map):
    # category_map: dictionary key --> list of objects
    # check primary object category: subjectivity, relationship, body
    subjectivity_words = category_map['lemmatized']['subjectivity']
    relationship_words = category_map['lemmatized']['relationship']
    body_words = category_map['lemmatized']['body']

    # for each object in each row, find the corresponding category
    df['object_category'] = df['object'].apply(lambda x: 'subjectivity' if x in subjectivity_words else ('relationship' if x in relationship_words else ('body' if x in body_words else 'NAN')))

    # check relational subjectivity, a standalone category
    relational_subjectivity_words = category_map['lemmatized']['relational_subjectivity']
    df['if_relational_subjectivity'] = df.apply(lambda x: 1 if x['object'] in relational_subjectivity_words else 0, axis=1)
    return df

## 4. dataset specific operations (this is where the specific processing steps are defined)
### congress
def year_to_congress(start_year, end_year):
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year.")

    # The first Congress began in 1789
    FIRST_CONGRESS_YEAR = 1789
    first_congress = (start_year - FIRST_CONGRESS_YEAR) // 2 + 1
    last_congress = (end_year - FIRST_CONGRESS_YEAR) // 2 + 1

    if first_congress < 1:
        first_congress = 1

    return list(range(first_congress, last_congress + 1))

def congress_to_year(congress):
    FIRST_CONGRESS_YEAR = 1789
    year = (congress - 1) * 2 + FIRST_CONGRESS_YEAR
    # correct 1949 to be 1950, as congress happens every two years
    if year ==1949:
        return 1950
    else:
        return year

def speechid_to_congress(speech_id):
    speech_id = str(speech_id)
    if len(speech_id)==9:
        congress_number = speech_id[:2]
    elif len(speech_id)==10:
        congress_number = speech_id[:3]
    return int(congress_number)

def speechid_to_congress_df_operation(df):
    # extract year from speech_id
    df['year'] = df['speech_id'].apply(lambda x: congress_to_year(speechid_to_congress(x)))
    return df

def congress_variable_rename(df):
    df['dataset']='congress'
    df.rename(columns={"speech_id":"doc_id"},inplace=True)
    return df

# generate age
def birthday_to_birthyear(birthday, year):
    if pd.isnull(birthday):
        return np.nan
    else:
        birth_year = int(birthday.split('-')[0])
        return birth_year

### movie
def movie_variable_rename(df):
    df['imdbID_tt'] = df['imdbID'].apply(lambda x: 'tt'+str(x).zfill(7))
    df['dataset']='movie'
    df.rename(columns={"imdbID_tt":"doc_id", "Year":"year"},inplace=True)
    return df

### caselaw
def caseid_to_year(df):
    df['year']= df['decision_date'].apply(lambda x: int(x.split('-')[0]))
    return df

def case_variable_rename(df):
    df['dataset']='caselaw'
    df.rename(columns={"case_id":"doc_id"},inplace=True)
    return df

### nyt
def nyt_variable_rename(df):
    df['dataset']='nyt'
    df.rename(columns={"GOID":"doc_id", "Year": "year"},inplace=True)
    return df

### fiction
def fiction_variable_rename(df):
    df['dataset']='fiction'
    df.rename(columns={"filename":"doc_id"},inplace=True)
    return df