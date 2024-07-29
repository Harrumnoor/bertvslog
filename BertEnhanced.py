import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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
    incorrect_samples = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Log incorrect cases
            incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                incorrect_samples.append((batch['question'][idx], batch['query'][idx], labels[idx].item(), preds[idx].item()))

    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)

    logging.info(f'{phase} Precision: {precision}')
    logging.info(f'{phase} Recall: {recall}')
    logging.info(f'{phase} F1 Score: {f1}')
    logging.info(f'{phase} Accuracy: {accuracy}')

    # Detailed log of incorrect predictions
    for question, query, true_label, predicted_label in incorrect_samples:
        logging.error(f'{phase} Misclassified - Question: {question}, Query: {query}, Predicted: {predicted_label}, Actual: {true_label}')

    return accuracy, true_labels, predictions

# Main execution
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to('cuda')

# Load and prepare data
data = load_data('correct_Jun1824.json', 'incorrect_Jun1824.json')

# Prepare for cross-validation
train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)  # Splitting for test data
train_data, val_data = train_test_split(train_val_data, test_size=0.11, random_state=42)  # Splitting remaining data for validation

train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)
test_dataset = SQLCorrectnessDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

best_val_accuracy = 0
early_stopping_counter = 0
early_stopping_patience = 3

for epoch in range(3):
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        total_train_accuracy += (preds == labels).cpu().numpy().mean()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    logging.info(f'Epoch {epoch+1} | Average Training Loss: {avg_train_loss} | Average Training Accuracy: {avg_train_accuracy}')

    val_accuracy, val_true_labels, val_predictions = validate_model(model, val_loader, 'Validation')

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        # Save the best model
        model_dir = './my_trained_model_v2'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            logging.info("Early stopping triggered")
            break

# Final validation
validate_model(model, val_loader, 'Final Validation')

# Testing
validate_model(model, test_loader, 'Test')

# Save the final model and tokenizer
model_dir = './my_trained_model_v2'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.config.save_pretrained(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
tokenizer.save_pretrained(model_dir)
