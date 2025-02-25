import accelerate
import transformers
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast

import os
import json
import pandas as pd
import numpy as np
import pickle
from  tqdm import tqdm
import argparse

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load config for each dataset
def load_config(dataset_name):
    with open("config_pb_classification.json", "r") as f:
        config = json.load(f)
    return config.get(dataset_name, {})

class ManageDataset(Dataset):
    def __init__(self, tokenizer, sentences, primary_labels, subcategory_labels, target_char_spans):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.primary_labels = primary_labels  # List of primary category labels
        self.subcategory_labels = subcategory_labels  # List of subcategory labels
        self.char_spans = target_char_spans  # List of character spans for target words

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Tokenize the sentence into BERT tokens with offset mappings
        inputs = self.tokenizer(
            self.sentences[idx],
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=256,
            return_offsets_mapping=True  # Return offset mappings for sub-token positions
        )

        # Generate the manag_mask
        manag_mask = self._get_manag_mask(
            self.sentences[idx],
            inputs["input_ids"][0],
            inputs["offset_mapping"][0],
            self.char_spans[idx]
        )

        # Return tokens' embeddings and the labels
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "manag_mask": manag_mask,
            "primary_labels": torch.tensor(self.primary_labels[idx], dtype=torch.long),
            "subcategory_labels": torch.tensor(self.subcategory_labels[idx], dtype=torch.long)
        }

    def _get_manag_mask(self, sentence, input_ids, offset_mapping, target_char_span):
        # Initialize manag_mask
        manag_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        # Iterate over BERT tokens and align with target word's character span
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                continue  # Skip special tokens like [CLS], [SEP], [PAD]
            if (start >= target_char_span[0] and start < target_char_span[1]) or \
               (end > target_char_span[0] and end <= target_char_span[1]) or \
               (start <= target_char_span[0] and end >= target_char_span[1]):
                manag_mask[i] = True
        return manag_mask


# Define the BERT classification model: 
# 3 primary categories and 10 subcategories
class BERTClassificationModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_primary_labels=3, num_subcategory_labels=10): 
        super(BERTClassificationModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Classification heads for primary and subcategories
        self.primary_classifier = nn.Linear(self.bert.config.hidden_size, num_primary_labels)
        self.subcategory_classifier = nn.Linear(self.bert.config.hidden_size, num_subcategory_labels)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)
        # Save the configuration
        self.config = self.bert.config
        self.num_primary_labels = num_primary_labels
        self.num_subcategory_labels = num_subcategory_labels

    def forward(self, input_ids, attention_mask, manag_mask):
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Apply manag_mask to get embeddings of target tokens
        manag_mask_expanded = manag_mask.unsqueeze(-1).expand(last_hidden_state.size())
        target_embeddings = last_hidden_state * manag_mask_expanded.float()

        # Compute average embeddings for each sample in the batch
        token_counts = manag_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, 1)
        # Avoid division by zero
        token_counts[token_counts == 0] = 1
        avg_embeddings = target_embeddings.sum(dim=1) / token_counts  # (batch_size, hidden_size)

        # Apply dropout
        pooled_output = self.dropout(avg_embeddings)

        # Get logits from classifiers
        primary_logits = self.primary_classifier(pooled_output)  # (batch_size, num_primary_labels)
        subcategory_logits = self.subcategory_classifier(pooled_output)  # (batch_size, num_subcategory_labels)

        return primary_logits, subcategory_logits

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        # Save configuration with label information
        self.config.num_primary_labels = self.num_primary_labels
        self.config.num_subcategory_labels = self.num_subcategory_labels
        self.config.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory):
        # Load the model configuration
        config = BertModel.from_pretrained(load_directory).config
        # Get the number of labels from the saved config
        num_primary_labels = config.num_primary_labels
        num_subcategory_labels = config.num_subcategory_labels
        # Initialize the model
        model = cls(
            bert_model_name=load_directory,
            num_primary_labels=num_primary_labels,
            num_subcategory_labels=num_subcategory_labels
        )
        # Load the model state dict
        model_load_path = os.path.join(load_directory, 'pytorch_model.bin')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_load_path))
            model = model.to('cuda')
        else:
            model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
        return model

def infer(sentences, char_spans, model, tokenizer, batch_size):
    dataset = ManageDataset(tokenizer, sentences, [0]*len(sentences), [0]*len(sentences), char_spans)
    loader = DataLoader(dataset, batch_size)  # Set batch size according to your needs

    model.eval()
    pred_primary_labels = []
    pred_subcategory_labels = []
    primary_confidences = []
    subcategory_confidences = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            manag_mask = batch['manag_mask'].to(device)

            # Forward pass
            primary_logits, subcategory_logits = model(input_ids, attention_mask, manag_mask)

            # Convert logits to probabilities using softmax
            primary_probs =torch.softmax(primary_logits, dim=1)
            subcategory_probs = torch.softmax(subcategory_logits, dim=1)

            # Get the predicted labels (indices of max probabilities)
            primary_preds = torch.argmax(primary_probs, dim=1)
            subcategory_preds = torch.argmax(subcategory_probs, dim=1)

            primary_confidence = torch.max(primary_probs, dim=1).values
            subcategory_confidence = torch.max(subcategory_probs, dim=1).values

            # Append predictions and confidences to the lists
            pred_primary_labels.extend(primary_preds.cpu().numpy())
            pred_subcategory_labels.extend(subcategory_preds.cpu().numpy())
            primary_confidences.extend(primary_confidence.cpu().numpy())
            subcategory_confidences.extend(subcategory_confidence.cpu().numpy())

    return pred_primary_labels, pred_subcategory_labels, primary_confidences, subcategory_confidences

def text_to_tuple(text):
    # convert string to tuple, and items to integers
    return tuple(map(int, text.strip('()').split(',')))

def single_file_processing_pipeline(df, if_csv, loaded_model, loaded_tokenizer, batch_size):

    # convert string to tuple, and items to integers
    if if_csv:
        df['focal_char_span'] = df['focal_char_span'].apply(text_to_tuple)    

    # Select df that passed the WSD filter; or that value is 999 (not in verb form)
    df_selected = df[(df['WSD_pred']>0)&(df['WSD_conf']>0.95)]

    # create a list of sentences
    sentences = df_selected['mgmt_sents'].tolist()
    sentence_idx = df_selected.index.tolist()
    char_spans = df_selected['focal_char_span'].tolist()

    # Run inference
    pred_primary_labels, pred_subcategory_labels, primary_confidences, subcategory_confidences = infer(sentences, char_spans, loaded_model, loaded_tokenizer, batch_size)

    # check the length of the predictions
    assert len(pred_primary_labels) == len(sentences), "Predictions and sentences length mismatch"

    df['pb_primary_predictions'] = 999.0
    df['pb_primary_confidences'] = 999.0
    df['pb_subcategory_predictions'] = 999.0
    df['pb_subcategory_confidences'] = 999.0

    # Create a temporary DataFrame with the new values
    updates = pd.DataFrame({'pb_primary_predictions': pred_primary_labels,
                            'pb_primary_confidences': primary_confidences,
                            'pb_subcategory_predictions': pred_subcategory_labels,
                            'pb_subcategory_confidences': subcategory_confidences},
                            index=sentence_idx)
    df.update(updates)

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
    loaded_tokenizer = BertTokenizerFast.from_pretrained(model_path)
    loaded_model = BERTClassificationModel.from_pretrained(model_path)
    loaded_model.to(device)
    
    # process data
    batch_size = config.get('batch_size', 100)
    if_csv = config.get('if_csv', False) # in some datasets, files are saved as csv, others as pickle
    for i in range(len(data_paths)):
        if if_csv:
            df = pd.read_csv(data_paths[i])
        else:
            df = pd.read_pickle(data_paths[i])
        df = single_file_processing_pipeline(df, if_csv, loaded_model, loaded_tokenizer, batch_size)
        if if_csv:
            df.to_csv(processed_paths[i], index=False)
        else:
            df.to_pickle(processed_paths[i])
    print(f"Finished processing {dataset_name} dataset")
    return

# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset for person-business management classification")
    args = parser.parse_args()
    process_dataset(args.dataset_name)
