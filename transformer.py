import os
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import torch.optim as optim

def prepare_val_list():
    val_files = []
    for i in range(1, 50):
        x = random.randint(1, 7028)
        if x not in val_files:
            val_files.append(x)
    val_files = list(map(lambda x: str(x) + '.txt', val_files))
    return val_files

def getData(dataPath, MAX_DOC_LEN, val=0):
    documentPath = f'{dataPath}/judgement'
    summaryPath = f'{dataPath}/summary'
    dataset = {'document': [], 'summary': []}
    count = 0
    for file in os.listdir(documentPath):
        count += 1
        if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
            continue
        doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
        doc_lines = [line.strip() for line in doc_in.readlines()]
        summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
        summ_lines = [line.strip() for line in summ_in.readlines()]
        if len(doc_lines) == 0 or len(summ_lines) == 0:
            continue
        if val == 0 and file not in val_files:
            dataset['document'].append(' '.join(doc_lines))
            dataset['summary'].append(' '.join(summ_lines))
        if val == 1 and file in val_files:
            dataset['document'].append(' '.join(doc_lines))
            dataset['summary'].append(' '.join(summ_lines))
    df = pd.DataFrame(dataset)
    return df

model_name = "nsi319/legal-led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataPath = "./dataset/IN-Abs"
val_files = prepare_val_list()
train_df = getData(f'{dataPath}/train-data')
val_df = getData(f'{dataPath}/train-data')

def get_test_data(dataPath):
    documentPath = f'{dataPath}/judgement'
    summaryPath = f'{dataPath}/summary'
    dataset = {'document': [], 'summary': []}
    count = 0
    for file in os.listdir(documentPath):
        count += 1
        if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
            continue
        doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
        doc_lines = [line.strip() for line in doc_in.readlines()]
        summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
        summ_lines = [line.strip() for line in summ_in.readlines()]
        if len(doc_lines) == 0 or len(summ_lines) == 0:
            continue
        dataset['document'].append(' '.join(doc_lines))
        dataset['summary'].append(' '.join(summ_lines))
    df = pd.DataFrame(dataset)
    return df

dataPath = "./dataset/IN-Abs"
test_df = get_test_data(f'{dataPath}/test-data')

input_seq_length = 10000
output_seq_length = 1000
vocab_size = tokenizer.vocab_size
embedding_dim = 768
hidden_units = 512
batch_size = 10
epochs = 2

document_texts = list(val_df['document'].values)
summary_texts = list(val_df['summary'].values)

document_sequences = tokenizer(document_texts, padding='max_length', truncation=True, max_length=input_seq_length, return_tensors="np", return_token_type_ids=False, return_attention_mask=False)['input_ids']
summary_sequences = tokenizer(summary_texts, padding='max_length', truncation=True, max_length=output_seq_length + 1, return_tensors="np", return_token_type_ids=False, return_attention_mask=False)['input_ids']

x = document_sequences
y = summary_sequences[:, :-1]

model_config = model.config
embedding_dim = model_config.hidden_size

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)
        return output

num_heads = 8
num_layers = 4

transformer_model = TransformerModel(vocab_size, embedding_dim
