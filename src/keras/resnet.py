import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Define the identity block
def identity_block(x, filters, kernel_size):
    f1, f2, f3 = filters

    # 1x1 Convolution
    x_shortcut = x
    x = Conv2D(f1, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 Convolution
    x = Conv2D(f2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 Convolution
    x = Conv2D(f3, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    # Add shortcut and final activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

# Define the convolutional block
def convolutional_block(x, filters, kernel_size, strides=(2, 2)):
    f1, f2, f3 = filters

    # Save the input value for the shortcut path
    x_shortcut = x

    # 1x1 Convolution with strides
    x = Conv2D(f1, (1, 1), strides=strides, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 Convolution
    x = Conv2D(f2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 Convolution
    x = Conv2D(f3, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    x_shortcut = Conv2D(f3, (1, 1), strides=strides, padding='valid')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    # Add shortcut and final activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

# Define a simplified ResNet for CIFAR-10
def ResNetCIFAR10(input_shape=(32, 32, 3), num_classes=10):
    input_layer = Input(shape=input_shape)

    # Initial Convolution layer
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First ResNet block (conv2_x)
    x = convolutional_block(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], kernel_size=(3, 3))

    # Second ResNet block (conv3_x)
    x = convolutional_block(x, filters=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2))
    x = identity_block(x, filters=[128, 128, 512], kernel_size=(3, 3))

    # Third ResNet block (conv4_x)
    x = convolutional_block(x, filters=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))

    # Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage with CIFAR-10
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create an instance of the simplified ResNet for CIFAR-10
    model = ResNetCIFAR10(input_shape=(32, 32, 3), num_classes=10)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')
