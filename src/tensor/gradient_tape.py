import tensorflow as tf

x1 = tf.Variable(3.)
x2 = tf.Variable(1., trainable=False)
with tf.GradientTape() as t:
    t.watch(x2)
    y = (x1 + 2 * x2) ** 2
dy_dx = t.gradient(y, [x1, x2])
print(f'dy/dx1 = {dy_dx[0]}')
print(f'dy/dx2 = {dy_dx[1]}')