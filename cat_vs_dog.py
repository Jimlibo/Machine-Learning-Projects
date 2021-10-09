# Convolutional Neural Network with Pretrained Model

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

keras = tf.keras


# 1. splitting the data manually to 80% train, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

get_label_name = metadata.features['label'].int2str


# just display 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


# 2.reshaping our images according to IMG_SIZE
IMG_SIZE = 160  # all images will be resized to 160x160 pixels


def format_example(format_image, format_label):
    """
    returns an image that is reshaped to IMG_SIZE
    """
    format_image = tf.cast(format_image, tf.float32)
    format_image = (format_image / 127.5) - 1
    format_image = tf.image.resize(format_image, (IMG_SIZE, IMG_SIZE))
    return format_image, format_label


# 3.creating our train, validation and test data
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)


# 4.shuffling and batching our images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# 5.creating the base model (pretrained model)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,        # this means we will put our own classifier at the top, and not the predefined one
    weights='imagenet'
)

# 6.freezing the base (keeping the same weights and not retraining the base_model)
base_model.trainable = False


# 7.adding our own classifier
global_layer = keras.layers.GlobalAveragePooling2D()   # this will transform the output of the base_model into 1D array
prediction_layer = keras.layers.Dense(1)  # since we have only 2 classes we need only 1 neuron for the prediction

model = keras.Sequential([   # we simply combine the base_model with the previous two layers
    base_model,
    global_layer,
    prediction_layer
])

# 8.compiling the model
base_learning_rate = 0.0001    # very small learning rate, so that the model won't have major changes

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),    # optimizer function
    loss=keras.losses.BinaryCrossentropy(from_logits=True),        # we use Binary because we have only 2 classes
    metrics=['accuracy']    # what output we are interested in
)


# 9.we can evaluate our model to see how it goes without any extra training
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print('Base loss:', loss0)
print('Base accuracy:', accuracy0)


# 10.training the model
history = model.fit(
    train_batches,
    epochs=initial_epochs,
    validation_data=validation_batches
)

accuracy = history.history['accuracy']
print("Final accuracy:", accuracy)


# 11. for such large models, we can save them in keras, so we don't have to train them again if we need them
model.save("dogs_vs_cats.h5")   # the .h5 format is for models
