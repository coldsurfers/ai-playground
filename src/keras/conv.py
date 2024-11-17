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