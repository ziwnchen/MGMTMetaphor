import pickle
import pandas as pd
import numpy as np
import random
import json
import argparse
import os
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import nltk # note that server without internet will need to append additional path
from nltk.corpus import words

import spacy
nlp = spacy.load("en_core_web_sm")

tqdm.pandas()
english_words = set(words.words())

# load config for each dataset
def load_config(dataset_name):
    with open("config_fillmask_preprocessing.json", "r") as f:
        config = json.load(f)
    return config.get(dataset_name, {})

# load config for target words
def load_target_words():
    with open("fillmask_target_objects.json", "r") as f:
        config = json.load(f)
    subjectivity_ls = config['raw']['subjectivity'] # raw: including plural forms, not lemmatized
    body_ls = config['raw']['body']
    relationship_ls = config['raw']['relationship']
    return subjectivity_ls, body_ls, relationship_ls

def lemmatize_words(words):
    # Create a dictionary to store the lemmas of the words
    lemmas = {}
    for word in words:
        doc = nlp(word)
        for token in doc:
            lemmas[word] = token.lemma_
    
    return lemmas

# load total object list
subjectivity_ls, body_ls, relationship_ls = load_target_words()
total_ls = subjectivity_ls+body_ls+relationship_ls
total_lemma_dict = lemmatize_words(total_ls)

# Step 1: text cleaning and intital filtering of sentences containing objects
# text cleaning functions
def replace_newlines(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    return text
    
def remove_quotation(text):
    # remove double quotation marks
    text = re.sub(r'\"', '', text)
    text = re.sub(r'\'{2}', '', text)
    return text

def remove_spaces(text):
    # remove more than one spaces
    text = re.sub(' +', ' ', text)
    return text

def text_cleaning(text):
    # overall text cleaning function
    if isinstance(text, float):
        # if text is nan
        return ""
    text = replace_newlines(text)
    text = remove_quotation(text)
    text = remove_spaces(text)
    return text

#  calculate sentence length
def sent_length(sent):
    return len(sent.split(" "))

# check ocr quality, this is a simple check that does not consider punctuations
def ocr_quality_check(text):
    tokens = text.split(" ")
    non_english_count = sum(1 for token in tokens if token not in english_words)
    total_words = len(tokens)
    if total_words == 0:
        return 1
    non_english_ratio = non_english_count / total_words
    return non_english_ratio 

# Initial Filtering
# extract sentences containing given objects, this would be a coarse filtering
# there might be chances that a given word is not in the sentence (eg, want parent but only get apparent)
def sentence_extract(text, object_ls):
    relevant_sents = []
    if isinstance(text, float) == False:
        sents = nltk.sent_tokenize(text)
        for sent in sents:
            sent_lower = sent.lower()
            for word in object_ls:
                if word in sent_lower:
                    relevant_sents.append((word,sent_lower))
    return relevant_sents

# Further Filtering
def mask_token_in_text(doc, token_id):
    masked_tokens = []
    for token in doc:
        if token.i == token_id:
            masked_tokens.append('[MASK]')
        else:
            masked_tokens.append(token.text)

    # Join the tokens back to form the masked sentence
    masked_text = ' '.join(masked_tokens)
    return masked_text

# Further filtering: remove sentences that did not contain the word we need
# also make sure that the word detected is a noun
def sentence_structure_check(word, text, lemma_dict):
    # Check lemma form
    word_lemma = lemma_dict[word]
    doc = nlp(text)

    matches = []
    for token in doc:
        if token.lemma_ == word_lemma and token.pos_ == "NOUN":  # Only select nouns
            focal_object = token.lemma_ #store the lemma form
            focal_object_id = token.i  # ID of the object
            head_verb = "NA"
            if_VO = False
            if_SV = False

            # Check structure: VO (Verb-Object)
            if token.dep_ in ['dobj', 'pobj', 'iobj'] and token.head.pos_ == "VERB":
                if_VO = True
                head_verb = token.head.lemma_

            # Check structure: SV (Subject-Verb)
            if 'subj' in token.dep_ and token.head.pos_ == "VERB":
                if_SV = True
                head_verb = token.head.lemma_

            # Mask the identified token
            object_masked_sent = mask_token_in_text(doc, focal_object_id)

            # Store result
            matches.append((focal_object, focal_object_id, object_masked_sent, if_VO, if_SV, head_verb))

    if matches:
        return True, matches  # List of all occurrences
    else:
        return False, []  # No matches found


# wrapper function to deal with list of (word, sent) structure
def apply_parallel(df_group):
    df_group['result'] = df_group['relevant_sent'].apply(lambda x:  sentence_structure_check(x[0],x[1], total_lemma_dict))
    return df_group

def parallelize_dataframe(df, func, n_chunks, ncores):
    pool = Pool(ncores)

    df_split = np.array_split(df, n_chunks)
    results = []

    with tqdm(total=len(df_split)) as pbar:
        for result in pool.imap_unordered(func, df_split):
            results.append(result)
            pbar.update(1)

    pool.close()
    pool.join()
    return pd.concat(results)
  
def df_formatting(df):
    df['if_selected'] = df['result'].apply(lambda x: x[0])
    df['matches'] = df['result'].apply(lambda x: x[1])
    df = df[df['if_selected']==True]
    df.reset_index(inplace=True, drop=True)

    # explode matches
    df = df.explode('matches')
    df['object'] = df['matches'].apply(lambda x: x[0])
    df['object_id'] = df['matches'].apply(lambda x: x[1])
    df['object_mask'] = df['matches'].apply(lambda x: x[2])
    df['if_VO'] = df['matches'].apply(lambda x: x[3])
    df['if_SV'] = df['matches'].apply(lambda x: x[4])
    df['head_verb'] = df['matches'].apply(lambda x: x[5])

    df['sent_unmask'] = df['relevant_sent'].apply(lambda x: x[1])
    df.drop(columns=['relevant_sent', 'result', 'matches'], inplace=True) # drop intermediate columns
    return df

# Overall Preprocessing Pipeline
def initial_filtering(df, text_col, total_obj_ls):
    # initial filtering
    print('Begin Initial Filtering')
    df[text_col]=df[text_col].apply(lambda x: text_cleaning(x))
    df['relevant_sent'] = df[text_col].apply(lambda x: sentence_extract(x,total_obj_ls))
    df.drop(columns=[text_col], inplace=True)

    df = df[df['relevant_sent'].astype(bool)]
    df = df.explode('relevant_sent').reset_index(drop=True)
    df['sent_length'] = df['relevant_sent'].apply(lambda x: sent_length(x[1]))
    df['noneng_ratio'] = df['relevant_sent'].apply(lambda x: ocr_quality_check(x[1]))
    df = df[(df['sent_length']>10) & (df['sent_length']<100) & (df['noneng_ratio']<0.5)]
    df.reset_index(inplace=True, drop=True)
    return df

def further_filter(df, doc_id, ncores, nchunks):
    df = parallelize_dataframe(df, apply_parallel, nchunks, ncores)
    df = df_formatting(df)
    # remove duplicates
    df.drop_duplicates(subset=[doc_id, 'object_mask', 'object','object_id'], inplace=True)
    df.reset_index(inplace=True, drop=True)    
    return df     

def flatten_path_dict(data_path_dict):
    total_paths = []
    parent_paths = data_path_dict.keys()
    for path in parent_paths:
        for file in data_path_dict[path]:
            total_paths.append(path + file)
    return total_paths

def concat_dfs(files, save_path, doc_id):
    total_df_lst = []
    for file in files:
        df = pd.read_pickle(file)
        total_df_lst.append(df)
        df['source'] = file.split("/")[-1].split(".")[0]
    total_df = pd.concat(total_df_lst, ignore_index=True)
    # drop duplicates (in case of cross file overlap)
    total_df.drop_duplicates(subset=[doc_id, 'object_mask', 'object','object_id'], inplace=True)

    # save
    if isinstance(save_path, str):
        total_df.to_pickle(save_path)
    elif isinstance(save_path, list):
        nparts = len(save_path)
        df_parts = np.array_split(total_df, nparts)
        for i in range(nparts):
            df_parts[i].to_pickle(save_path[i])
    print(f"Finished concatenating {len(files)} files")
    return

# process dataset, use the config file to specify parameters
def process_dataset(dataset_name):
    # load dataset config
    config = load_config(dataset_name)
    ncores = config.get('ncores', cpu_count())
    nchunks = config.get('nchunks', ncores*10)
    data_paths = flatten_path_dict(config['data_paths'])
    processed_paths = flatten_path_dict(config['processed_paths'])
    assert len(data_paths) == len(processed_paths), "Error: Number of data files and processed files should be the same"

    text_col = config['text_col']
    doc_id = config['doc_id']
    if_concat = config.get('if_concat', False)
    for i in range(len(data_paths)):
        print(f"Processing {data_paths[i]}")
        df = pd.read_csv(data_paths[i])
        df = initial_filtering(df, text_col, total_ls)
        df = further_filter(df, doc_id, ncores, nchunks)
        df.to_pickle(processed_paths[i]) # because the processed data contains tuples
    if if_concat:
        concat_dfs(processed_paths, config['concat_path'], doc_id)
    
    print(f"Finished processing {dataset_name} dataset")

# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to preprocess for fillmask operations")
    args = parser.parse_args()
    process_dataset(args.dataset_name)

