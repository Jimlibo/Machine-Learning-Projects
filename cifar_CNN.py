# Convolutional Neural Networks


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# 1.load our dataset and create name for each label
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']


# 2.normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Let's have a look at one of these images
# index = 1
# plt.imshow(train_images[index], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[index][0]])
# plt.show()
############################################


# 3.creating the part of the model that recognizes the different patterns
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))    # first convolutional layer (1)
model.add(layers.MaxPooling2D((2, 2)))                                              # maxpool on first layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))                             # second convolutional layer (2)
model.add(layers.MaxPooling2D((2, 2)))                                              # maxpool on second layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))                             # third convolutional layer (3)


# 4.create neural_network to classify the previous patterns'
model.add(layers.Flatten())                         # inside input layer
model.add(layers.Dense(64, activation='relu'))      # inside hidden layer
model.add(layers.Dense(10))                         # inside output layer


# 5. compile the model
model.compile(optimizer='adam',    # optimizer function
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),    # loss/cost function
              metrics=['accuracy'])   # what output we are interested in


# 6.training the model
history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))


# 7.evaluating the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_accuracy)
print('Test loss:', test_loss)
