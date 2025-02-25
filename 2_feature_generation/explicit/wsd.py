import accelerate
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast, BertConfig,AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F  # Import the functional module to apply softmax

import pandas as pd
import os
import json
from tqdm import tqdm
import argparse

# load config for each dataset
def load_config(dataset_name):
    with open("config_wsd.json", "r") as f:
        config = json.load(f)
    return config.get(dataset_name, {})

# functions
class ManageDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, target_char_spans):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.char_spans = target_char_spans

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Tokenize the sentence into BERT tokens with offset mappings (fast tokenizer)
        inputs = self.tokenizer(
            self.sentences[idx],
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=256,
            return_offsets_mapping=True # return tuple indicating the sub-token's start position
        )

        # Generate the manag_mask
        manag_mask = self._get_manag_mask(
            self.sentences[idx],
            inputs["input_ids"][0],
            inputs["offset_mapping"][0],
            self.char_spans[idx]
        )

        # Return tokens' embeddings and the label
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "manag_mask": manag_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
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

class BERTWSDModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(BERTWSDModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)
        # Save the configuration
        self.config = self.bert.config
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, manag_mask):
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

        # Get logits from classifier
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        return logits

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        self.config.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory):
        # Load the model configuration
        config = BertModel.from_pretrained(load_directory).config
        # Initialize the model
        model = cls(bert_model_name=load_directory)
        # Load the model state dict
        model_load_path = os.path.join(load_directory, 'pytorch_model.bin')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_load_path))
            model = model.to('cuda')
        else:
            model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
        return model

def infer(sentences, char_spans, model, tokenizer, batch_size, device):
    dataset = ManageDataset(tokenizer, sentences, [0]*len(sentences), char_spans)  # Dummy labels just for data processing
    loader = DataLoader(dataset, batch_size)  # Set batch size according to your needs

    model.eval()
    pred_labels = []
    confidences = []  # To store prediction confidences

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            manag_mask = batch["manag_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, manag_mask)

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=1)

            # Get the predicted labels and their corresponding confidences
            preds = torch.argmax(logits, dim=1)
            conf = probs[range(probs.shape[0]), preds].tolist()  # Get the confidence of the predicted class for each sample

            pred_labels.extend(preds.tolist())
            confidences.extend(conf)

    return pred_labels, confidences  # Return both predicted labels and their confidences

def text_to_tuple(text):
    # convert string to tuple, and items to integers
    return tuple(map(int, text.strip('()').split(',')))

def single_file_processing_pipeline(df, if_csv, loaded_model, loaded_tokenizer, batch_size, device):

    # convert string to tuple, and items to integers
    if if_csv:
        df['focal_char_span'] = df['focal_char_span'].apply(text_to_tuple)

    # select verb sentences: wsd is only for verbs, as the noun form of management is not ambiguous
    df_verbs = df[df['focal_pos'].apply(lambda x: x.startswith('V'))]

    # select verb sentences
    verb_sent = list(df_verbs['mgmt_sents'])
    verb_sent_idx = list(df_verbs.index)
    verb_char_spans = list(df_verbs['focal_char_span'])
    
    print("Total Number of Sentences", len(verb_sent))
    predictions, confidence_scores = infer(verb_sent, verb_char_spans, loaded_model, loaded_tokenizer, batch_size, device)

    # ensure the mapping
    assert len(verb_sent)==len(predictions)
        
    df['WSD_pred'] = 999.0
    df['WSD_conf'] = 999.0

    # Create a temporary DataFrame with the new values
    updates = pd.DataFrame({'WSD_pred': predictions, 'WSD_conf': confidence_scores}, index=verb_sent_idx)
    df.update(updates)

    return df

def flatten_path_dict(data_path_dict):
    total_paths = []
    parent_paths = data_path_dict.keys()
    for path in parent_paths:
        for file in data_path_dict[path]:
            total_paths.append(path + file)
    return total_paths

def process_dataset(dataset_name, device):
    # load dataset config
    config = load_config(dataset_name)
    model_path = config['model_path']
    data_paths = flatten_path_dict(config['data_paths'])
    processed_paths = flatten_path_dict(config['processed_paths'])
    assert len(data_paths) == len(processed_paths), "Error: Number of data files and processed files should be the same"

    # load model
    loaded_tokenizer = BertTokenizerFast.from_pretrained(model_path)
    loaded_model = BERTWSDModel.from_pretrained(model_path)
    loaded_model.to(device)
    
    # process data
    batch_size = config.get('batch_size', 100)
    if_csv = config.get('if_csv', False) # in some datasets, files are saved as csv, others as pickle
    for i in range(len(data_paths)):
        if if_csv:
            df = pd.read_csv(data_paths[i])
        else:
            df = pd.read_pickle(data_paths[i])
        df = single_file_processing_pipeline(df, if_csv, loaded_model, loaded_tokenizer, batch_size, device)
        if if_csv:
            df.to_csv(processed_paths[i], index=False)
        else:
            df.to_pickle(processed_paths[i])
    print(f"Finished processing {dataset_name} dataset")
    return


# load data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset for WSD operations")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_dataset(args.dataset_name, device)