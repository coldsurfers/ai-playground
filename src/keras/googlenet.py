import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Define the Inception module
def inception_module(x, filters):
    (f1, f3_r, f3, f5_r, f5, f_pool) = filters

    # 1x1 Convolution Branch
    branch1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    # 1x1 followed by 3x3 Convolution Branch
    branch2 = Conv2D(f3_r, (1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(f3, (3, 3), padding='same', activation='relu')(branch2)

    # 1x1 followed by 5x5 Convolution Branch
    branch3 = Conv2D(f5_r, (1, 1), padding='same', activation='relu')(x)
    branch3 = Conv2D(f5, (5, 5), padding='same', activation='relu')(branch3)

    # 3x3 Pooling followed by 1x1 Convolution Branch
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = Conv2D(f_pool, (1, 1), padding='same', activation='relu')(branch4)

    # Concatenate all branches
    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    return x

# Define the GoogLeNet model
def GoogLeNet(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = Input(shape=input_shape)

    # Initial convolutional and pooling layers
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = inception_module(x, [384, 192, 384, 48, 128, 128])

    # Average Pooling, Dropout, and Fully Connected Layer
    x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(input_layer, output_layer)

    return model

# Example usage
if __name__ == "__main__":
    # Create an instance of GoogLeNet
    model = GoogLeNet(input_shape=(224, 224, 3), num_classes=1000)

    # Define an exponential decay learning rate schedule
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    # Compile the model with the optimizer using the learning rate schedule
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create some dummy data for demonstration purposes
    x_train = np.random.random((16, 224, 224, 3))  # 16 sample training dataset
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 1000, 16), num_classes=1000)

    # Fit the model (for demonstration purposes, training for a small number of epochs)
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=4)
