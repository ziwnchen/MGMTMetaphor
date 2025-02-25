import accelerate
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
from collections import Counter
import re
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

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

class BERTClassificationModel(nn.Module):
    def __init__(self, num_primary_labels, num_subcategory_labels, bert_model_name='bert-base-uncased'):
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

## Train the model
def train_val_split(df, val_size=0.1):
    sentences = df["mgmt_sents"].tolist()
    primary_labels = df["primary"].tolist()
    secondary_labels = df["secondary"].tolist()
    target_char_spans = list(df['focal_char_span'])
    train_sentences, val_sentences, train_primary_labels, val_primary_labels, train_secondary_labels, val_secondary_labels, train_char_spans, val_char_spans = train_test_split(
        sentences, primary_labels, secondary_labels, target_char_spans, test_size=val_size, random_state=42, stratify=primary_labels)
    return train_sentences, val_sentences, train_primary_labels, val_primary_labels, train_secondary_labels, val_secondary_labels, train_char_spans, val_char_spans

def compute_class_weights(labels_list):
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    num_classes = len(label_counts)
    
    # Initialize weights
    class_weights = np.zeros(num_classes)
    
    for label in label_counts:
        class_weights[label] = total_samples / (num_classes * label_counts[label])
    
    # Normalize weights (optional)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

# Load the dataset
data_directory = "/zfs/projects/faculty/amirgo-management/COHA_data/processed_data/"
df=pd.read_pickle(data_directory+"coha_mgmt_sent_chatgpt_tagged_oct30_processed.pkl")
df_selected = df[(df['is_intransitive']==False)]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_sentences, val_sentences, train_primary_labels, val_primary_labels, train_secondary_labels, val_secondary_labels, train_char_spans, val_char_spans = train_val_split(df_selected)

train_dataset = ManageDataset(tokenizer, train_sentences, train_primary_labels, train_secondary_labels, train_char_spans)
val_dataset = ManageDataset(tokenizer, val_sentences, val_primary_labels, val_secondary_labels, val_char_spans)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Compute class weights for primary categories
primary_class_weights = compute_class_weights(train_primary_labels)
primary_class_weights = torch.tensor(primary_class_weights, dtype=torch.float)

# Compute class weights for subcategories
subcategory_class_weights = compute_class_weights(train_secondary_labels)
subcategory_class_weights = torch.tensor(subcategory_class_weights, dtype=torch.float)

# Initialize the model
num_primary_labels = len(set(train_primary_labels))  # Number of unique primary labels
num_subcategory_labels = len(set(train_secondary_labels))  # Number of unique subcategory labels
print("Num of Primary Label", num_primary_labels)
print("Num of Subcategory Label", num_subcategory_labels)
model = BERTClassificationModel(
    num_primary_labels=num_primary_labels,
    num_subcategory_labels=num_subcategory_labels,
    bert_model_name='bert-base-uncased'
)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion_primary = nn.CrossEntropyLoss(weight=primary_class_weights.to(device))
criterion_subcategory = nn.CrossEntropyLoss(weight=subcategory_class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Number of training epochs
num_epochs = 8

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Training Phase
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        manag_mask = batch['manag_mask'].to(device)
        primary_labels = batch['primary_labels'].to(device)
        subcategory_labels = batch['subcategory_labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        primary_logits, subcategory_logits = model(input_ids, attention_mask, manag_mask)
        
        # Compute losses with class weights
        loss_primary = criterion_primary(primary_logits, primary_labels)
        loss_subcategory = criterion_subcategory(subcategory_logits, subcategory_labels)
        
        # Total loss
        total_loss = loss_primary + loss_subcategory
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss:.4f}")
    
    # Evaluation Phase
    model.eval()
    all_primary_preds = []
    all_primary_labels = []
    all_subcategory_preds = []
    all_subcategory_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            manag_mask = batch['manag_mask'].to(device)
            primary_labels = batch['primary_labels'].to(device)
            subcategory_labels = batch['subcategory_labels'].to(device)
            
            # Forward pass
            primary_logits, subcategory_logits = model(input_ids, attention_mask, manag_mask)
            
            # Get predictions
            primary_preds = torch.argmax(primary_logits, dim=1)
            subcategory_preds = torch.argmax(subcategory_logits, dim=1)
            
            # Collect results
            all_primary_preds.extend(primary_preds.cpu().numpy())
            all_primary_labels.extend(primary_labels.cpu().numpy())
            all_subcategory_preds.extend(subcategory_preds.cpu().numpy())
            all_subcategory_labels.extend(subcategory_labels.cpu().numpy())
    
    # Compute metrics for primary categories
    primary_accuracy = accuracy_score(all_primary_labels, all_primary_preds)
    primary_precision, primary_recall, primary_f1, _ = precision_recall_fscore_support(
        all_primary_labels, all_primary_preds, average='weighted')
    
    # Compute metrics for subcategories
    subcategory_accuracy = accuracy_score(all_subcategory_labels, all_subcategory_preds)
    subcategory_precision, subcategory_recall, subcategory_f1, _ = precision_recall_fscore_support(
        all_subcategory_labels, all_subcategory_preds, average='weighted')
    
    print(f"Primary Category - Accuracy: {primary_accuracy:.4f}, Precision: {primary_precision:.4f}, Recall: {primary_recall:.4f}, F1-score: {primary_f1:.4f}")
    print(f"Subcategory - Accuracy: {subcategory_accuracy:.4f}, Precision: {subcategory_precision:.4f}, Recall: {subcategory_recall:.4f}, F1-score: {subcategory_f1:.4f}")
    print("\n")

# Save the model
save_directory = "/zfs/projects/faculty/amirgo-management/BERT/PB_MultiClass_Full_Oct30/"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)