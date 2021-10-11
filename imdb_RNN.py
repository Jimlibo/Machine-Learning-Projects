# Natural Language Processing with RNN
# We train a sentiment analysis model, using
# the imdb dataset from keras


import tensorflow as tf
import tensorflow.keras as keras
from keras.datasets import imdb  # the dataset we will use
from keras.preprocessing import sequence
import os
import numpy as np


VOCAB_SIZE = 88584    # num of different words
MAX_LEN = 250
BATCH_SIZE = 64

# loading our data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# formatting all data to have exactly 250 charcters
train_data = sequence.pad_sequences(train_data, MAX_LEN)
test_data = sequence.pad_sequences(test_data, MAX_LEN)

# create the model
model = tf.keras.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, 32),    # layer for the word embedding
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')  # we use only one node because we have only positive/negative
])

# compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# training the model
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)


# evaluating the model
results = model.evaluate(test_data, test_labels)
print(results)


# making predictions on our own text
word_index = imdb.get_word_index()


def encode_text(text):    # function to convert words to encoded integers
    tokens = keras.preprocessing.text.text_to_word_sequence(text)  # splitting all the words
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]  # word -> right integer
    return sequence.pad_sequences([tokens], MAX_LEN)[0]


def predict(text):    # a function that predicts the sentiment from a given review
    enc_text = encode_text(text)
    pred = np.zeros((1, 250))       # our model expect an argument of dimensions N x 250
    pred[0] = enc_text             # the only line of the pred table is our review
    result = model.predict(pred)
    return result

# just for testing the model given a review
positive_review = "That movie was so awesome! I loved it from the very first time"
negative_review = "The movie was absolutely terrible! The worst thing i have ever seen"

pos_result = predict(positive_review)
print(pos_result[0])

neg_result = predict(negative_review)
print(neg_result[0])








