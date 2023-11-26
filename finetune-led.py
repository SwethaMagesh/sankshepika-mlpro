
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

tokenizer.vocab_size
tokenizer.decode(tokens)
tokenizer.embedding

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


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Attention, Dense

# Define model parameters
vocab_size = tokenizer.vocab_size  # adjust based on your vocabulary size
lstm_units = 512
max_input_length = 10000  # adjust based on your input document length
max_output_length = 1000  # adjust based on your output summary length
embedding_dim = 768
batch_size = 10
epochs = 2



# Define the model architecture
input_layer = Input(shape=(max_input_length,))

# Embedding layer for input
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

# Bidirectional LSTM encoder
encoder = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(embedding_layer)

# Attention mechanism
attention = Attention()([encoder, encoder])

# Bidirectional LSTM decoder
decoder = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(attention)

# Fully connected layer for output
output_layer = Dense(units=vocab_size, activation='softmax')(decoder)

# Create the model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

one_hot_encoded_y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

embedding_dim = 768
hidden_units = 512
batch_size = 10
epochs = 2
model.fit(x=x, y=one_hot_encoded_y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
