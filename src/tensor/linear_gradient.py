import tensorflow as tf
import numpy as np

x = tf.constant([1.,3.,5.,7.])
y = tf.constant([2.,3.,4.,5.])
w = tf.Variable(1.)
b = tf.Variable(0.5)
learning_rate = 0.01
epochs = 1000

def train_step(x, y):
    with tf.GradientTape() as t:
        y_hat = w * x + b
        loss = (y_hat - y) ** 2
        grads = t.gradient(loss, [w, b])
        w.assign_sub(learning_rate * grads[0])
        b.assign_sub(learning_rate * grads[1])

for i in range(epochs):
    for k in range(len(y)):
        train_step(x[k], y[k])

print('w: {:8.5f}   b: {:8.5f}'.format(w.numpy(), b.numpy()))

f = 'x: {:8.5f} --> y: {:8.5f}'
for k in range(len(y)):
    y_hat = w * x[k] + b
    print(f.format(x[k].numpy(), y_hat.numpy()))