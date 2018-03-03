#https://colab.research.google.com/notebooks/mlcc/creating_and_manipulating_tensors.ipynb?hl=zh-cn#scrollTo=PT1sorfH-DdQ
import tensorflow as tf
# Create a graph.
g = tf.Graph()

print('------------------矢量加法-----------------------------------------------')
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
  # Create another six-element vector. Each element in the vector will be initialized to 1. The first argument is the shape of the tensor (more on shapes below).
  ones = tf.ones([6], dtype=tf.int32)
  # Add the two vectors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)
  # Create a session to run the default graph.
  with tf.Session() as sess:
      print(just_beyond_primes.eval())
print('------------------张量形状-------------------------------------------------------------')
with tf.Graph().as_default():
  # A scalar (0-D tensor).
  scalar = tf.zeros([])
  # A vector with 3 elements.
  vector = tf.zeros([3])
  # A matrix with 2 rows and 3 columns.
  matrix = tf.zeros([2, 3])
  with tf.Session() as sess:
    print ('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval())
    print ('vector has shape', vector.get_shape(), 'and value:\n', vector.eval())
    print ('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval())
print('---------------与之前一样的张量加法，不过使用的是广播：-----------------------------------')
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
  # Create a constant scalar with value 1.
  ones = tf.constant(1, dtype=tf.int32)
  # Add the two tensors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)
  with tf.Session() as sess:
    print (just_beyond_primes.eval())
print('--------------------矩阵乘法-----------------------------------------------------------')
with tf.Graph().as_default():
  # Create a matrix (2-d tensor) with 3 rows and 4 columns.
  x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],dtype=tf.int32)
  # Create a matrix with 4 rows and 2 columns.
  y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)
  # Multiply `x` by `y`.
  # The resulting matrix will have 3 rows and 2 columns.
  matrix_multiply_result = tf.matmul(x, y)
  with tf.Session() as sess:
    print (matrix_multiply_result.eval())