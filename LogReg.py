import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import json
import logging
from tqdm import tqdm
import os
import numpy as np

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='training_log.log', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data function
def load_data(correct_file_path, incorrect_file_path):
    with open(correct_file_path, 'r') as file:
        correct_data = json.load(file)
    with open(incorrect_file_path, 'r') as file:
        incorrect_data = json.load(file)
    for item in correct_data:
        item['correctness'] = 1
    for item in incorrect_data:
        item['correctness'] = 0
    return correct_data + incorrect_data

# Dataset class
class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_input = f"{item['question']} [SEP] {item['query']}"
        inputs = self.tokenizer.encode_plus(
            combined_input,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(item['correctness'], dtype=torch.long),
            'question': item['question'],
            'query': item['query']
        }

# Model validation with detailed error logging
def validate_model(model, loader, phase='Validation'):
    model.eval()
    predictions = []
    true_labels = []
    confidence_scores = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = probs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidence_scores.extend(probs.max(dim=1).values.cpu().numpy())

    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)
    log_loss_score = log_loss(true_labels, confidence_scores)

    logging.info(f'{phase} Precision: {precision}')
    logging.info(f'{phase} Recall: {recall}')
    logging.info(f'{phase} F1 Score: {f1}')
    logging.info(f'{phase} Accuracy: {accuracy}')
    logging.info(f'{phase} Log Loss: {log_loss_score}')

    return accuracy, true_labels, predictions, confidence_scores

# Load and prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_dir = './my_trained_model_v2'
model = BertForSequenceClassification.from_pretrained(model_dir).to('cuda')

data = load_data('correct_Jun1824.json', 'incorrect_Jun1824.json')

# Prepare for cross-validation
train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.11, random_state=42)

train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)
test_dataset = SQLCorrectnessDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Validate BERT model
validate_model(model, val_loader, 'Validation')
validate_model(model, test_loader, 'Test')

# Train logistic regression model
vectorizer = TfidfVectorizer(max_features=5000)
log_reg_model = make_pipeline(vectorizer, LogisticRegression())

# Prepare data for logistic regression
def prepare_log_reg_data(data):
    texts = [f"{item['question']} [SEP] {item['query']}" for item in data]
    labels = [item['correctness'] for item in data]
    return texts, labels

train_texts, train_labels = prepare_log_reg_data(train_data)
val_texts, val_labels = prepare_log_reg_data(val_data)
test_texts, test_labels = prepare_log_reg_data(test_data)

log_reg_model.fit(train_texts, train_labels)

# Validate logistic regression model
def validate_log_reg_model(model, texts, labels, phase='Validation'):
    predictions = model.predict(texts)
    confidence_scores = model.predict_proba(texts).max(axis=1)

    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    log_loss_score = log_loss(labels, confidence_scores)

    logging.info(f'{phase} Precision: {precision}')
    logging.info(f'{phase} Recall: {recall}')
    logging.info(f'{phase} F1 Score: {f1}')
    logging.info(f'{phase} Accuracy: {accuracy}')
    logging.info(f'{phase} Log Loss: {log_loss_score}')

    return accuracy, labels, predictions, confidence_scores

validate_log_reg_model(log_reg_model, val_texts, val_labels, 'Validation (Logistic Regression)')
validate_log_reg_model(log_reg_model, test_texts, test_labels, 'Test (Logistic Regression)')

# Compare models
bert_val_accuracy, _, _, bert_val_confidence_scores = validate_model(model, val_loader, 'Validation')
log_reg_val_accuracy, _, _, log_reg_val_confidence_scores = validate_log_reg_model(log_reg_model, val_texts, val_labels, 'Validation (Logistic Regression)')

logging.info(f"BERT Validation Accuracy: {bert_val_accuracy}, Average Confidence Score: {np.mean(bert_val_confidence_scores)}")
logging.info(f"Logistic Regression Validation Accuracy: {log_reg_val_accuracy}, Average Confidence Score: {np.mean(log_reg_val_confidence_scores)}")

bert_test_accuracy, _, _, bert_test_confidence_scores = validate_model(model, test_loader, 'Test')
log_reg_test_accuracy, _, _, log_reg_test_confidence_scores = validate_log_reg_model(log_reg_model, test_texts, test_labels, 'Test (Logistic Regression)')

logging.info(f"BERT Test Accuracy: {bert_test_accuracy}, Average Confidence Score: {np.mean(bert_test_confidence_scores)}")
logging.info(f"Logistic Regression Test Accuracy: {log_reg_test_accuracy}, Average Confidence Score: {np.mean(log_reg_test_confidence_scores)}")
