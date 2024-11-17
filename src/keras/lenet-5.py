from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Layer
from keras import backend as K
import numpy as np

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