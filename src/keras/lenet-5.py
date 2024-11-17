import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Layer, Input
from tensorflow.keras import backend as K
import numpy as np

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = np.pad(
    x_train,
    ((0, 0,), (2, 2), (2, 2), (0, 0)), 'constant'
)
x_test = np.pad(
    x_test,
    ((0, 0), (2, 2), (2, 2), (0, 0)),
    'constant'
)
x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(
            name='mu',
            shape=(int(input_shape[1]), self.units),
            initializer='uniform',
            trainable=True
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

model = Sequential()
model.add(
    Input(shape=(32, 32, 1))
)
model.add(
    Conv2D(
        6,
        kernel_size=(5, 5),
        activation='tanh',
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)
model.add(
    Conv2D(
        16,
        kernel_size=(5, 5),
        activation='tanh'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(RBFLayer(10, 0.5))

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs=20,
    verbose=1,
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test)
print('accuracy:', score[1])