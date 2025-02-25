import accelerate
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax

import pandas as pd
import numpy as np
import pickle
from  tqdm import tqdm
import os
import json
import argparse

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load config for each dataset
def load_config(dataset_name):
    with open("config_fillmask.json", "r") as f:
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

# functions for processing prediction objects
# convert token to idx assigned by tokenizer
def token_to_idx(token, tokenizer):
    return tokenizer.encode(token, add_special_tokens=False)

# filter out non-bert vocab
def gen_token_idx(total_objects, tokenizer):
    filtered_objects = []
    bert_vocab = set(tokenizer.get_vocab().keys())
    for word in total_objects:
        token_ids = token_to_idx(word, tokenizer)
        if word in bert_vocab and len(token_ids) == 1: # Ensure that the word is in the BERT vocabulary and is not split into multiple tokens
            filtered_objects.append(word)
    filtered_objects_idx = [token_to_idx(token, tokenizer)[0] for token in filtered_objects]
    return filtered_objects, filtered_objects_idx

# check missing in each list aftering filtering
def check_missing(original_objects, predicted_objects, random_objects, filtered_objects):
    missing_original = set(original_objects) - set(filtered_objects)
    missing_predicted = set(predicted_objects) - set(filtered_objects)
    missing_random = set(random_objects) - set(filtered_objects)
    print(f"Missing {len(missing_original)} out of {len(original_objects)} original objects")
    print(f"Missing {len(missing_predicted)} out of {len(predicted_objects)} predicted objects")
    print(f"Missing {len(missing_random)} out of {len(random_objects)} random objects")

# dataframe processing for fillmask
# data processing: extract mask indices, prepare dataset conversion
def data_preprocessing(df, col, tokenizer, max_length):
    selected_indices = []
    processed_data = []

    for idx, row in df.iterrows():
        sentence = row[col]
        inputs = tokenizer(sentence, max_length=max_length, truncation=True, return_tensors="pt")
        if tokenizer.mask_token_id in inputs['input_ids']: # Ensure that the mask token is present
            mask_indices = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist() # Find the indices of the mask token
            processed_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'mask_indices': mask_indices,
            })
            selected_indices.append(idx)

    return processed_data, selected_indices

class MaskedSentenceDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'mask_indices': self.data[idx]['mask_indices'],
        }

# padding to align sentence tensor
def custom_collate_fn(batch):
    input_ids = [item['input_ids'].squeeze(0) for item in batch]  # Ensure each is 1D
    attention_masks = [item['attention_mask'].squeeze(0) for item in batch]  # Ensure each is 1D
    mask_indices = [item['mask_indices'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True)

    return {'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'mask_indices': mask_indices}

# return probability of certain tokens
def batch_predict_with_dataloader(model, dataset, tokenizer, target_entities, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    model.eval()
    total_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = softmax(outputs.logits, dim=-1) #the shape of the predictions tensor: should be [batch size, token length of padded sequence, vocabulary]
            for i in range(len(input_ids)):
                sequence_target_predictions = []
                for mask_index in mask_indices[i]:
                    masked_predictions = predictions[i, mask_index]
                    # select prediction of certain idx
                    target_predictions = masked_predictions[target_entities]
                    sequence_target_predictions.append(target_predictions)
                total_predictions.append(sequence_target_predictions)

    return total_predictions

def convert_to_df(predictions, selected_indices, df):
    # Initialize the new columns as lists
    df['object_target_pred_prob'] = [[] for _ in range(len(df))]

    assert len(predictions) == len(selected_indices), "Error: Number of predictions and selected indices should be the same"

    predicted_probs = [p[0].tolist() for p in predictions]
    updates = pd.DataFrame({'object_target_pred_prob': predicted_probs}, index=selected_indices)
    df.update(updates)

    return df

## higher level functions: process dataset and single files
def single_file_processing_pipeline(df, total_object_idx, model, tokenizer, tokenizer_max_len, batch_size):
    processed_data, selected_indices = data_preprocessing(df, 'object_mask', tokenizer, tokenizer_max_len) # data processing, mostly truncation
    masked_sentence_dataset = MaskedSentenceDataset(processed_data) # convert to a dataset
    predictions = batch_predict_with_dataloader(model, masked_sentence_dataset, tokenizer, total_object_idx, batch_size) # run batch prediction
    df = convert_to_df(predictions, selected_indices, df) # format prediction to df
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
    loaded_model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)

    # load original objects and target objects
    original_objects, predicted_objects, random_objects = load_target_words()
    total_objects = original_objects + predicted_objects + random_objects
    total_objects, total_object_idx = gen_token_idx(total_objects, loaded_tokenizer)    # filter out non-bert vocab
    check_missing(original_objects, predicted_objects, random_objects, total_objects)   # check missing words after filtering

    ## Run batch prediction
    tokenizer_max_len = config.get('tokenizer_max_len', 128)
    batch_size = config.get('batch_size', 100)
    if_csv = config.get('if_csv', False) # in some datasets, files are saved as csv, others as pickle
    for i in range(len(data_paths)):
        print(f"Processing {data_paths[i]}")
        if if_csv:
            df = pd.read_csv(data_paths[i])
        else:
            df = pd.read_pickle(data_paths[i])
        df = single_file_processing_pipeline(df, total_object_idx, loaded_model, loaded_tokenizer, tokenizer_max_len, batch_size)
        # save to pkl (due to the list format of object_target_pred_prob)
        df.to_pickle(processed_paths[i])

    print(f"Finished processing {dataset_name} dataset")
    return

# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset for fillmask prediction operations")
    args = parser.parse_args()
    process_dataset(args.dataset_name)
