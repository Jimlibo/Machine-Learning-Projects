# Classification


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


# create a model that given an iris flower, it finds out what exactly type it is

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']   # features
SPECIES = ['Setosa', 'Versicolor', 'Virginica']   # three types of iris flowers

# 1.downloads the files in the computer and provides a path to those files
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# 2.we create the dataframes with all the info in them
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)   # for training the model
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)     # for testing the model

# 3.we split the features from the labels
y_train = train.pop('Species')
y_test = test.pop('Species')


# function for creating appropriate input
def input_fn(in_feature, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(in_feature), labels))  # creates a dataset from a panda dataframe

    # if we are training the model then we shuffle and repeat
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# 4.creating the feature columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# 5.building the model
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],   # we have two hidden layers of 30 and 10 nodes respectively
    n_classes=3)            # we have 3 possible classes

# 6.training the model
classifier.train(
    input_fn=lambda: input_fn(train, y_train, training=True),  # we need to pass as argument a function object
    steps=5000)  # steps are similar to epochs

# 7.evaluating the model
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, y_test, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))  # prints the accuracy of our model


# 8.for specific predictions
def input_fn(in_features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(in_features)).batch(batch_size)  # returns a dataset without labels


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print('Please input numeric values as prompted:\n')   # get feature values of a specific flower
for feature in features:
    valid = True
    while valid:
        val = input(feature + ":")
        if not val.isdigit():
            valid = False
        predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))  # predict the class of given flower
for element in predictions:
    class_id = element['class_ids'][0]  # find the class
    probability = element['probabilities'][class_id]  # find the probability of belonging to class_id

    print('\nPrediction is: "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))  # prints the results
