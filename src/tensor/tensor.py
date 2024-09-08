# tensor
# - same type
# - scalar
# - rank0 tensor, rank1 tensor, rank2 tensor
import tensorflow as tf
import numpy as np

a = tf.constant(10.)
b = tf.constant([1, 2, 3, 4])
c = tf.constant([  [  # 첫 번째 깊이 (Depth 0)    [1, 2, 3, 4],   # 첫 번째 행
    [5, 6, 7, 8],   # 두 번째 행
    [9, 10, 11, 12] # 세 번째 행
  ],

  [  # 두 번째 깊이 (Depth 1)    [13, 14, 15, 16], # 첫 번째 행
    [17, 18, 19, 20], # 두 번째 행
    [21, 22, 23, 24]  # 세 번째 행
  ]
], dtype=tf.float32)

print(a.dtype, '\n', a)
print(b.shape, '\n', b)
print(c.device)

x = tf.Variable(10.)
y = tf.Variable([[1.,2.,3.], [4.,5.,6.]])
z = np.array([[1.,3.], [2.,4.], [3.,5.]], dtype=np.float32)
print(x.dtype, x)
print(y.shape, y)
print(y.device)