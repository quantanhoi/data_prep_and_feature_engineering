import os

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Task: predict which website an article comes from based on its title
LOGDIR = "./text_models"
DATA_DIR = "./Data"
DATASET_NAME = "titles_full.csv"
TITLE_SAMPLE_PATH = os.path.join(DATA_DIR, DATASET_NAME)
COLUMNS = ['title', 'source']   # load the 2 columns of the dataset into a DataFrame with key names 'title' and 'source'

titles_df = pd.read_csv(TITLE_SAMPLE_PATH, header=None, names=COLUMNS)
titles_df.head(100)


tokenizer = Tokenizer() # Tokenizer creates a vocabulary mapping words → integers
tokenizer.fit_on_texts(titles_df.title) # fit_on_texts() builds the vocabulary from all titles
integerized_titles = tokenizer.texts_to_sequences(titles_df.title)  # texts_to_sequences() converts each title to a list of integers



integerized_titles[:3]


token_id = integerized_titles[0][0]
word = tokenizer.index_word[token_id]         # maps ID → word
print(f"{token_id} → {word}")

# to decode a full sequence back into words:
decoded = [tokenizer.index_word[id] for id in integerized_titles[0]]  # this create a list of words from the first title's integerized sequence
print(decoded)


VOCAB_SIZE = len(tokenizer.index_word) # Total unique words
VOCAB_SIZE


DATASET_SIZE = tokenizer.document_count # Total number of titles
DATASET_SIZE


MAX_LEN = max(len(sequence) for sequence in integerized_titles) # Longest title
MAX_LEN


def create_sequences(texts, max_len=MAX_LEN):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences,
                                     max_len,
                                     padding='post')
    return padded_sequences
""" IMPORTANT NOTE:
Neural networks need fixed-size inputs
pad_sequences() adds zeros to make all sequences the same length
padding='post' adds zeros at the end
"""


sample_titles = create_sequences(["holy cash cow  batman - content is back",
                                 "close look at a flu outbreak upends some common wisdom"])
sample_titles


CLASSES = {
    'github': 0,
    'nytimes': 1,
    'techcrunch': 2
}
N_CLASSES = len(CLASSES)


def encode_labels(sources):
    classes = [CLASSES[source] for source in sources]
    one_hots = utils.to_categorical(classes)
    return one_hots

""" 
Converts source labels to one-hot vectors:
'github' → [1, 0, 0]
'nytimes' → [0, 1, 0]
'techcrunch' → [0, 0, 1]
"""



# Determine how many samples to use for training (80% of the total dataset)
N_TRAIN = int(DATASET_SIZE * 0.8)
print("Number of training samples: " + str(N_TRAIN))

# Load the CSV of titles and sources into a DataFrame
# – header=None    : the file has no header row
# – names=COLUMNS : assign our ['title', 'source'] column names
titles_df = pd.read_csv(TITLE_SAMPLE_PATH, header=None, names=COLUMNS)

# Slice out the first N_TRAIN rows for the training split
# – titles_train  : the text inputs (titles) for training
# – sources_train : the labels (source names) for training
titles_train, sources_train = (
    titles_df.title[:N_TRAIN],
    titles_df.source[:N_TRAIN]
)

# Slice the remaining rows for validation
# – titles_valid  : the text inputs for validation
# – sources_valid : the labels for validation
titles_valid, sources_valid = (
    titles_df.title[N_TRAIN:],
    titles_df.source[N_TRAIN:]
)



X_train, Y_train = create_sequences(titles_train), encode_labels(sources_train)
X_valid, Y_valid = create_sequences(titles_valid), encode_labels(sources_valid)



	
def build_dnn_model(embed_dim):

    model = models.Sequential([
        # Embedding Layer:
        #   Input:  (batch_size, MAX_LEN)         — sequences of word indices
        #   Output: (batch_size, MAX_LEN, embed_dim) — sequences of dense vectors
        #   Learns a lookup table mapping each word index → embed_dim-length vector
        layers.Embedding(VOCAB_SIZE + 1,
                         embed_dim,
                         input_shape=[MAX_LEN]),

        # Lambda Layer (Average Pooling):
        #   Input:  (batch_size, MAX_LEN, embed_dim)
        #   Output: (batch_size, embed_dim)
        #   Averages over the time dimension to get one fixed-size vector per sample
        layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),

        # Dense Layer:
        #   Input:  (batch_size, embed_dim)
        #   Output: (batch_size, N_CLASSES)
        #   Softmax activation to produce a probability distribution over classes
        layers.Dense(N_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',                 # Adam optimizer
        loss='categorical_crossentropy',  # loss for multi-class classification
        metrics=['accuracy']              # track accuracy during training
    )
    return model



titles_train[0], X_train[0], Y_train[0], sources_train[0]




tf.random.set_seed(33)

MODEL_DIR = os.path.join(LOGDIR, 'dnn')
#shutil.rmtree(MODEL_DIR, ignore_errors=True)

BATCH_SIZE = 300
EPOCHS = 100
EMBED_DIM = 10
PATIENCE = 0

dnn_model = build_dnn_model(embed_dim=EMBED_DIM)

dnn_history = dnn_model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_valid, Y_valid),
    callbacks=[callbacks.EarlyStopping(patience=PATIENCE),
               callbacks.TensorBoard(MODEL_DIR)],
)

pd.DataFrame(dnn_history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(dnn_history.history)[['accuracy', 'val_accuracy']].plot()

dnn_model.summary()