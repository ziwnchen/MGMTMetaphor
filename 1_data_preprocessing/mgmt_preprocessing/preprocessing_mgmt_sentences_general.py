import pandas as pd
import numpy as np
import os
import re
import json
import argparse
import pickle
from tqdm import tqdm
import nltk
from nltk import pos_tag
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
spacy.require_cpu() # use CPU for spacy, disable GPU warning
nlp = spacy.load("en_core_web_sm") 

tqdm.pandas()
lemmatizer = WordNetLemmatizer()
english_words = set(words.words())

# load config for each dataset
def load_config(dataset_name):
    with open("config_mgmt_preprocessing.json", "r") as f:
        config = json.load(f)
    return config.get(dataset_name, {})

# Step 1: text cleaning and intital filtering of management related sentences
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

## below are text cleaning functions not used, but can be useful for some cases
def remove_special_chars(text):
    # remove special characters, keep punctuations and numbers
    text = re.sub(r'[^A-Za-z0-9\s\.\,\!\?\:\;\-\']+', '', text)
    return text

def remove_html_tags(text):
    # remove html tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_heading(text):
    # remove heading in COHA dataset
    clean = re.sub(r'@@\d+\n\n', '', text)
    return clean

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

# extract sentences that contain "*manage*" or "*managing"
def sentence_extract(text):
    relevant_sents = []
    if isinstance(text, float) == False: # if text is not nan
        text = text.lower()
        if "manag" in text:
            sents = nltk.sent_tokenize(text)
            for sent in sents:
                if "manage" in sent or "managing" in sent:
                    relevant_sents.append(sent)                
    return relevant_sents

# get the character spans of words in a sentence, this is useful for highlighting management related words
# mainly used to extract contextual embeddings in the later stage
def get_word_char_spans(sentence, words):
    char_spans = []
    current_pos = 0
    for word in words:
        pattern = re.escape(word)
        match = re.search(pattern, sentence[current_pos:])
        if match is None:
            raise ValueError(f"Word '{word}' not found in sentence.")
        start_idx = current_pos + match.start()
        end_idx = current_pos + match.end()
        char_spans.append((start_idx, end_idx))
        current_pos = end_idx
    return char_spans

# extract management related words with POS tags
def extract_mgmt_with_pos(sentence):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    mgmt_word_loc = []
    char_spans = get_word_char_spans(sentence, words)
    
    for i in range(len(words)):
        word = words[i]
        tag = pos_tags[i]
        char_span = char_spans[i]
        if_intransitive = False
        if "manag" in word:
            # check if intransitve verb
            if "VB" in tag[1]:
                # check if the next word is "to"
                if i+1 < len(words) and words[i+1] == "to":
                    if_intransitive = True
            mgmt_word_loc.append((word, tag[1], char_span, if_intransitive))
    return mgmt_word_loc


# Step 2: extract management domains using semantic parsing
# for nouns, look for modifiers of management related words
# for verbs, look for objects of management related words
# Step 2: extract domains using semantic parsing
def extraction_noun_modifier(doc, word, char_span):
    for token in doc:
        start_idx = token.idx
        if token.text == word and start_idx == char_span[0]:
            children_text = []
            children = [child for child in token.children]
            children_text.extend([child.text for child in children if child.pos_ =="NOUN"])
            for child in token.children:
                if child.dep_ == "prep":
                    grand_children = [grand_child for grand_child in child.children]
                    children_text.extend([grand_child.text for grand_child in grand_children if grand_child.pos_ =="NOUN"])
            return children_text
    return []

def extract_subject_object(doc, word, char_span):
    for token in doc:
        start_idx = token.idx
        if token.text == word and start_idx == char_span[0]:
            subjects = []
            objects = []
            subject_text = "NA"
            object_text = "NA"
            for child in token.children:
                if child.dep_  == "nsubj":
                    subjects.append(child)
                elif child.dep_  in ["nsubjpass","dobj","pobj"]:
                    objects.append(child)
                elif child.dep_ == "agent":
                    subjects.extend([grandchild for grandchild in child.children])
            return subjects, objects
    return [], []

# noun parsing pipeline: select sentences with management noun and extract modifiers
def noun_extraction_pipeline(df):
    # select sentences with management noun
    df_noun = df[df['focal_pos'].apply(lambda x: x.startswith('NN'))].copy()
    noun_sent_idx = df_noun.index
    
    # noun extraction
    df_noun['modifier'] = df_noun.apply(lambda x: extraction_noun_modifier(nlp(x['mgmt_sents']), x['focal_word'], x['focal_char_span']), axis=1)

    # update the original df
    df['modifier'] = [[] for i in range(len(df))]
    df.loc[noun_sent_idx, 'modifier'] = df_noun['modifier']
    return df

#  verb parsing pipeline: select sentences with management verb and extract subjects and objects
def verb_extraction_pipeline(df):
    # select sentences with management verb
    df_verb = df[df['focal_pos'].apply(lambda x: x.startswith('VB'))].copy()
    verb_sent_idx = df_verb.index

    # verb extraction
    df_verb['subjects'], df_verb['objects'] = zip(*df_verb.apply(lambda x: extract_subject_object(nlp(x['mgmt_sents']), x['focal_word'], x['focal_char_span']), axis=1))

    # update the original df
    df['subjects'] = [[] for i in range(len(df))]
    df['objects'] = [[] for i in range(len(df))]
    df.loc[verb_sent_idx, 'subjects'] = df_verb['subjects']
    df.loc[verb_sent_idx, 'objects'] = df_verb['objects']

    return df

# Overall Preprocessing Pipeline
def initial_filtering(df, text_col):
    df[text_col]=df[text_col].apply(lambda x: text_cleaning(x))
    df['mgmt_sents']=df[text_col].apply(lambda x: sentence_extract(x))
    df.drop(columns=[text_col], inplace=True)
    
    df = df[df['mgmt_sents'].astype(bool)]
    df = df.explode('mgmt_sents').reset_index(drop=True)
    df['sent_length'] = df['mgmt_sents'].apply(sent_length)
    df['noneng_ratio'] = df['mgmt_sents'].apply(ocr_quality_check)
    # keep only sentences with length between 10 and 100, and non-english ratio less than 0.5
    df = df[(df['sent_length']>10) & (df['sent_length']<100) & (df['noneng_ratio']<0.5)] 
    df.reset_index(inplace=True, drop=True)
    
    # add pos tagging and focal word
    df['mgmt_tag'] = df['mgmt_sents'].progress_apply(lambda x: extract_mgmt_with_pos(x))
    df = df.explode('mgmt_tag').reset_index(drop=True)
    df['focal_word'] = df['mgmt_tag'].apply(lambda x: x[0])
    df['focal_pos'] = df['mgmt_tag'].apply(lambda x: x[1])
    df['focal_char_span'] = df['mgmt_tag'].apply(lambda x: x[2])
    df['if_intransitive'] = df['mgmt_tag'].apply(lambda x: x[3])
    df.drop(columns=['mgmt_tag'], inplace=True)
    
    return df

def further_filter(df):
    df = noun_extraction_pipeline(df)
    df = verb_extraction_pipeline(df)
    df['modifier'] = df['modifier'].apply(lambda x: " ".join(x))
    df['subjects'] = df['subjects'].apply(lambda x: " ".join([i.text for i in x]))
    df['objects'] = df['objects'].apply(lambda x: " ".join([i.text for i in x]))
    return df

def concat_dfs(files, save_path):
    total_df_lst = []
    for file in files:
        df = pd.read_pickle(file)
        total_df_lst.append(df)
        df['source'] = file.split("/")[-1].split(".")[0]
    total_df = pd.concat(total_df_lst, ignore_index=True)
    # save
    total_df.to_pickle(save_path)
    print(f"Finished concatenating {len(files)} files")
    return

def flatten_path_dict(data_path_dict):
    total_paths = []
    parent_paths = data_path_dict.keys()
    for path in parent_paths:
        for file in data_path_dict[path]:
            total_paths.append(path + file)
    return total_paths

# process dataset, use the config file to specify parameters
def process_dataset(dataset_name):
    config = load_config(dataset_name)
    # flatten the dictionary
    data_paths = flatten_path_dict(config['data_paths'])
    processed_paths = flatten_path_dict(config['processed_paths'])
    assert len(data_paths) == len(processed_paths), "Error: Number of data files and processed files should be the same"

    text_col = config['text_col']
    if_concat = config.get('if_concat', False)
    for i in range(len(data_paths)):
        print(f"Processing {data_paths[i]}")
        df = pd.read_csv(data_paths[i])
        df = initial_filtering(df, text_col)
        df = further_filter(df)
        df.to_pickle(processed_paths[i]) # because the processed data contains tuples
    if if_concat:
        concat_dfs(processed_paths, config['concat_path'])
    
    print(f"Finished processing {dataset_name} dataset")

# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to preprocess for management sentences")
    args = parser.parse_args()
    process_dataset(args.dataset_name)