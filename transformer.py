
# !pip install transformers
# !pip install sentencepiece
"""# Dict or simplify complex words
# Extractive method
# Abstractive LED finetune
# Abstractive Legal Pegasus with chunking Finetune
# Combine with prompt engineering

# LEGAL LED pretrained
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM






"""#Download dataset n extract"""

# !wget https://zenodo.org/records/7152317/files/dataset.zip?download=1
# !unzip '/content/dataset.zip?download=1'

"""## Preprocess the data into dataframe"""

import os
import random
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def prepare_val_list():
    val_files = []  # Add the validation files to be used
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
train_df = getData( f'{dataPath}/train-data')
val_df = getData(f'{dataPath}/train-data')

train_df

#  TEST DATA
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
test_df = get_test_data( f'{dataPath}/test-data')

len(train_df.iloc[0]['document'].split(' '))
len(train_df.iloc[0]['summary'].split(' '))

"""
# Finetune/ Custom model for LEGAL LED"""



"""## TOKENISE X N Y"""

input  = train_df.iloc[0]['document']

model_name = "nsi319/legal-led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens = tokenizer.encode(input)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.config.hidden_size


"""## max seq leng in our dataset"""

max = 0
wlist = []
for i in range(len(train_df)):
  input  = train_df.iloc[i]['summary']
  w = len(input.split(' '))
  wlist.append(w)

len(val_df['document'].values)

# Hyperparameters
input_seq_length = 10000  # Maximum input sequence length
output_seq_length = 1000 # Expected summary length
vocab_size = tokenizer.vocab_size  # Your vocabulary size
embedding_dim = 768
hidden_units = 512
batch_size = 10
epochs = 2

# Tokenize the text data
document_texts = list(val_df['document'].values)
summary_texts = list(val_df['summary'].values)

# Assuming you have document_texts and summary_texts as lists
document_sequences = tokenizer(document_texts, padding='max_length', truncation=True, max_length=input_seq_length, return_tensors="np", return_token_type_ids=False, return_attention_mask=False)['input_ids']
summary_sequences = tokenizer(summary_texts, padding='max_length', truncation=True, max_length=output_seq_length+1, return_tensors="np", return_token_type_ids=False, return_attention_mask=False)['input_ids']

# Create x and y
x = document_sequences
y = summary_sequences[:, :-1]



"""## Train custom model"""

# Access the model's configuration
model_config =  model.config

# Get the embedding dimension (hidden size)
embedding_dim = model_config.hidden_size



# Define model parameters
vocab_size = tokenizer.vocab_size  # adjust based on your vocabulary size
lstm_units = 512
max_input_length = 10000  # adjust based on your input document length
max_output_length = 1000  # adjust based on your output summary length
embedding_dim = 768
batch_size = 10
epochs = 2



# model.fit(x=x, y=one_hot_encoded_y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
import torch
import torch.nn as nn
import torch.optim as optim

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

# Model parameters

num_heads = 8
num_layers = 4

# Create the PyTorch Transformer model
transformer_model = TransformerModel(vocab_size, embedding_dim, hidden_units, num_heads, num_layers)

# Display the model architecture
print(transformer_model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

# Convert data to PyTorch tensors (x and y should be PyTorch tensors)
x_torch = torch.from_numpy(x)
y_hot = torch.nn.functional.one_hot(y, vocab_size)

y_torch = torch.from_numpy(y_hot)

# Training loop
for epoch in range(epochs):
    # Forward pass
    output = transformer_model(x_torch)
    
    # Compute the loss
    loss = criterion(output.view(-1, vocab_size), y_torch.view(-1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
