import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data to range [-1, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = (x_train - 0.5) * 2.0
x_test = (x_test - 0.5) * 2.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define AlexNet architecture (adjusted for CIFAR-10)
def create_alexnet():
    model = models.Sequential()

    # First Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Third, Fourth, and Fifth Convolutional Layers
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output from convolutional layers
    model.add(layers.Flatten())

    # First Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Second Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer (CIFAR-10 has 10 classes)
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Instantiate and compile the model
alexnet_model = create_alexnet()
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
alexnet_model.summary()

# Train the model
history = alexnet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = alexnet_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")
