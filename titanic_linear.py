# Linear Regression


from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
from six.moves import urllib
from IPython.display import clear_output
import tensorflow as tf


def make_input_fn(data_df, label_df, num_of_epochs=10, shuffle=True, batch_size=32):
    def input_function():   # inner function, that will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))    # create tf.data.Dataset object
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_of_epochs)  # split in batches of 32 and repeat process for number of epochs
        return ds   # return a  batch of the dataset
    return input_function   # return a function object for use


df_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')   # for training the model
df_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')    # for testing the model
y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()   # gets a list of all uniquevalues from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


train_input_fn = make_input_fn(df_train, y_train)
eval_input_fn = make_input_fn(df_eval, y_eval, num_of_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)  # creating the model

linear_est.train(train_input_fn)  # training the model
result = linear_est.evaluate(eval_input_fn)   # get model metrics/results by testing it on testing data

clear_output()  # clears the console from any unwanted output
print(result['accuracy'])    # the result variable is simply a dict of stats about our model

prediction = list(linear_est.predict(eval_input_fn))  # make a list with all the predictions

# find the chance of survival of entry <entry>
entry = 0
print(df_eval.loc[entry])
print('Chance of survival:', prediction[entry]['probabilities'][1])
