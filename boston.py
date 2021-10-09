# A simple model that predicts the stock price of houses in Boston.
# We took the dataset from the build-in datasets of keras, and used
# a Sequential model instead of a linear Classifier.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt         # for visual presentation of our model's predictions
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# loading our dataset from keras
boston = tf.keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston.load_data()

# normalizing function to make more accurate predictions
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_data))

# building the model
model = tf.keras.Sequential([     # the model has mae in [3.4, 4.0], if we want to further minimize it
    normalizer,                   # we tweak the layers , the num_of_neurons, etc. as we see fit
    layers.Dense(16),
    layers.Dense(4),
    layers.Dropout(0.2),
    layers.Dense(1)
])


# compiling the model
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mse'])     # we want to see mean absolute and squared error

# training the model
history = model.fit(train_data, train_labels,
                    epochs=100,
                    verbose=0,    # if we want to see the progress for each epoch, set verbose to 1
                    validation_split=0.5)

mae, mse = model.evaluate(test_data, test_labels, verbose=2)
print('Absolute Mean Error:', mae)     # around 3.4 - 4.0


# Plot predictions.
test_predictions = model.predict(test_data).flatten()


plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (stock price)')
plt.ylabel('Predictions (stock price)')
lims = [0, 60]    # stock prices are in space [0K, 60K]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()




