import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train data size: %d" %train_images.shape[0])
print("test data size: %d" %test_images.shape[0])
print("image size: %d X %d" %(train_images.shape[1], train_images.shape[2]))
print("answer example: %s" %str(train_labels[:20]))
print("train image example: \n%s" %str(train_images[1]))

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[1], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

train_images = train_images.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(
        kernel_size=(3, 3),
        filters=16,
        activation='relu'
    ),
    keras.layers.MaxPool2D(
        strides=(2, 2)
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        128,
        activation=tf.nn.relu
    ),
    keras.layers.Dense(
        10,
        activation=tf.nn.softmax
    )
])

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

def plot_loss(history):
    plt.figure(figsize=(16, 10))
    val = plt.plot(
        history.epoch,
        history.history['val_loss'],
        '--',
        label='Test'
    )
    plt.plot(
        history.epoch,
        history.history['loss'],
        color=val[0].get_color(),
        label='Train'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.xlim([0, max(history.epoch)])

plot_loss(history)