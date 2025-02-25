import pandas as pd
import numpy as np
import os
import time
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import json
import argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer # Load pretrained tokenizer

# load config for each dataset
def load_config(dataset_name):
    with open("config_measure_syn_adj.json", "r") as f:
        config = json.load(f)
    return config.get(dataset_name, {})

# load config for fillmask target words
def load_target_words():
    with open("fillmask_target_objects.json", "r") as f:
        config = json.load(f)
    
    # original words
    original_objects = []
    for key in ['subjectivity', 'body', 'relationship']: #exclude relational subjectivity, which is a redundant subset of subjectivity
        original_objects.extend(config['lemmatized'][key])

    # predicted words
    predicted_objects = []
    for key in config['prediction']:
        predicted_objects.extend(config['prediction'][key])

    # random words
    random_objects = config['random']['MacBERTh'] # MacBERTh is the model name

    return original_objects, predicted_objects, random_objects

# load the config again, for the categories of the target objects
def load_pred_syn_dictionary():
    with open("fillmask_target_objects.json", "r") as f:
        config = json.load(f)
    return config['prediction'], config['syn_group']

# set up private categories and their synonyms
def gen_synonym_dict(synonym_list):
    synonym_dict = {}
    for word in synonym_list:
        if type(word) == str:
            synonym_dict[word] = [word]
        else:
            for w in word:
                synonym_dict[w] = word
    return synonym_dict

# load total object list, filter out non-bert vocab
def gen_token_idx(total_objects, tokenizer):
    filtered_objects = []
    bert_vocab = set(tokenizer.get_vocab().keys())
    for word in total_objects:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if word in bert_vocab and len(token_ids) == 1: # Ensure that the word is in the BERT vocabulary and is not split into multiple tokens
            filtered_objects.append(word)
    return filtered_objects

# locate the index of the token in the selected objects predicted in the fillmask process (~290 o)
def token_to_idx(token, total_objects):
    return total_objects.index(token)

# convert index to token
def idx_to_token(idx, total_objects):
    return total_objects[idx]

def get_orig_object_prob(row, total_objects):
    obj = row['object']
    prob = row['object_target_pred_prob']
    return prob[token_to_idx(obj, total_objects)]

def get_group_prob(prob, words, total_objects):
    group_prob = []
    for word in words:
        if word not in total_objects:
            continue
        idx = token_to_idx(word, total_objects)
        group_prob.append(prob[idx])
    return np.mean(group_prob)

# group average of synonyms (if any)
def syn_group_prob(row, syn_dict, total_objects):
    obj = row['object']
    prob = row['object_target_pred_prob']
    if obj not in syn_dict:
        return prob[token_to_idx(obj, total_objects)]
    else:
        syns = syn_dict[obj]
        return get_group_prob(prob, syns, total_objects)

def gen_subgroup_ratio(df, pred_obj_dict, total_objects):
    pred_categories = pred_obj_dict.keys()
    for cat in pred_categories:
        df[f'{cat}_group_prob'] = df['object_target_pred_prob'].apply(lambda x: get_group_prob(x, pred_obj_dict[cat], total_objects))
    return df

# numerator: top mgmt group's average probability
def top_subgroup_ratio(row, pred_obj_dict):
    # find max subgroup
    subgroup_prob = []
    pred_categories = list(pred_obj_dict.keys())
    for cat in pred_categories:
        subgroup_prob.append(row[f'{cat}_group_prob'])
    
    max_idx = np.argmax(subgroup_prob)
    max_prob = subgroup_prob[max_idx]
    max_group = pred_categories[max_idx]

    denominator_prob1 = row['orig_prob']
    denominator_prob2 = row['orig_syn_prob']
    subgroup_orig_ratio = np.log(max_prob/denominator_prob1)
    subgroup_orig_syn_ratio = np.log(max_prob/denominator_prob2)
    return subgroup_orig_ratio, subgroup_orig_syn_ratio, max_group

## higher level functions: process dataset and single files
def single_file_processing_pipeline(df, total_objects, pred_obj_dict, syn_dict):
    df = df[df['object_target_pred_prob'].astype(bool)] # remove cases where no prediction is made, which is likely due to missing [MASK] in the text
    df.reset_index(inplace=True, drop=True) 

    df = df[df['object'].isin(total_objects)] # cannot calculate original object's probability if it is not in the total object list
    df['orig_prob'] = df.apply(lambda x: get_orig_object_prob(x, total_objects), axis = 1) # get the original object's probability
    df['orig_syn_prob'] = df.apply(lambda x: syn_group_prob(x, syn_dict, total_objects), axis = 1) # get the synonym group's probability
    df = gen_subgroup_ratio(df, pred_obj_dict, total_objects) # get the average probability of each subgroup
    df['subgroup_orig_ratio'], df['subgroup_orig_syn_ratio'], df['top_subgroup'] = zip(*df.progress_apply(lambda x: top_subgroup_ratio(x, pred_obj_dict), axis = 1)) # get the ratio of the top subgroup to the original synonym group

    # remove object_target_pred_prob
    df.drop(columns = ['object_target_pred_prob'], inplace = True)
    return df

def flatten_path_dict(data_path_dict):
    total_paths = []
    parent_paths = data_path_dict.keys()
    for path in parent_paths:
        for file in data_path_dict[path]:
            total_paths.append(path + file)
    return total_paths

def process_dataset(dataset_name):
    # load dataset config
    config = load_config(dataset_name)
    model_path = config['model_path']
    data_paths = flatten_path_dict(config['data_paths'])
    processed_paths = flatten_path_dict(config['processed_paths'])
    assert len(data_paths) == len(processed_paths), "Error: Number of data files and processed files should be the same"

    # load model
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load original objects and target objects
    original_objects, predicted_objects, random_objects = load_target_words()
    total_objects = original_objects + predicted_objects + random_objects
    total_objects = gen_token_idx(total_objects, loaded_tokenizer)    # filter out non-bert vocab
    pred_obj_dict, syn_group = load_pred_syn_dictionary() # load the categories of the target objects
    syn_dict = gen_synonym_dict(syn_group) # generate the synonym
    # filter out non-bert vocab for each category in the pred_obj_dict
    for key in pred_obj_dict:
        pred_obj_dict[key] = [word for word in pred_obj_dict[key] if word in total_objects]

    ## calculate measures for each file
    for i in range(len(data_paths)):
        print(f"Processing {data_paths[i]}")
        df = pd.read_pickle(data_paths[i])
        df = single_file_processing_pipeline(df, total_objects, pred_obj_dict, syn_dict)
        df.to_pickle(processed_paths[i])

    print(f"Finished processing {dataset_name} dataset")
    return

# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to generate implicit measures.")
    args = parser.parse_args()
    process_dataset(args.dataset_name)
