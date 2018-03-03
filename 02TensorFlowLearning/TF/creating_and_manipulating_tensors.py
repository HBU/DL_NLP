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
# 张量变形
# 由于张量加法和矩阵乘法均对运算数施加了限制条件，TensorFlow 编程者肯定会频繁改变张量的形状。
# 您可以使用 tf.reshape 方法改变张量的形状。 例如，您可以将 8x2 张量变形为 2x8 张量或 4x4 张量：
print('--------------------张量变形--------------------------------------------')
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)
  # Reshape the 8x2 matrix into a 2x8 matrix.
  reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
  # Reshape the 8x2 matrix into a 4x4 matrix
  reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])
  with tf.Session() as sess:
    print( "Original matrix (8x2):")
    print( matrix.eval())
    print( "Reshaped matrix (2x8):")
    print( reshaped_2x8_matrix.eval())
    print( "Reshaped matrix (4x4):")
    print( reshaped_4x4_matrix.eval())
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)
  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])
  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])
  with tf.Session() as sess:
    print  (  "Original matrix (8x2):")
    print  (  matrix.eval())
    print  (  "Reshaped 3-D tensor (2x2x4):")
    print  (  reshaped_2x2x4_tensor.eval())
    print  (  "1-D vector:")
    print  (  one_dimensional_vector.eval())

print('----------------------# 练习 1：改变两个张量的形状，使其能够相乘。------------------------------')
# 下面两个矢量无法进行矩阵乘法运算：
# a = tf.constant([5, 3, 2, 7, 1, 4])
# b = tf.constant([4, 6, 3])
# 请改变这两个矢量的形状，使其成为可以进行矩阵乘法运算的运算数。 然后，对变形后的张量调用矩阵乘法运算。
with tf.Graph().as_default(), tf.Session() as sess:
  # Task: Reshape two tensors in order to multiply them
  # Here are the original operands, which are incompatible for matrix multiplication:
  a = tf.constant([5, 3, 2, 7, 1, 4])
  b = tf.constant([4, 6, 3])
  # We need to reshape at least one of these operands so that the number of columns in the first operand equals the number of rows in the second operand.
  # Reshape vector "a" into a 2-D 2x3 matrix:
  reshaped_a = tf.reshape(a, [2, 3])
  # Reshape vector "b" into a 2-D 3x1 matrix:
  reshaped_b = tf.reshape(b, [3, 1])
  # The number of columns in the first matrix now equals
  # the number of rows in the second matrix. Therefore, you
  # can matrix mutiply the two operands.
  c = tf.matmul(reshaped_a, reshaped_b)
  print(c.eval())
  # An alternate approach: [6,1] x [1, 3] -> [6,3]

  print('----------------------  变量、初始化和赋值--------------------------')
  # g = tf.Graph()
  with g.as_default():
    # Create a variable with the initial value 3.
    v = tf.Variable([3])
    # Create a variable of shape [1], with a random initial value, # sampled from a normal distribution with mean 1 and standard deviation 0.35.
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
    #print(w.eval()) #未初始化，不能用
  print('------TensorFlow 的一个特性是变量初始化不是自动进行的。例如，以下代码块会导致错误：------------')
  with g.as_default():
    with tf.Session() as sess:
      try:
        v.eval()
      except tf.errors.FailedPreconditionError as e:
        print( "Caught expected error: ", e)
  print('----------初始化变量---最简单的方式是调用--global_variables_initializer-------------------------')
  with g.as_default():
    with tf.Session() as sess:
      initialization = tf.global_variables_initializer()
      sess.run(initialization)
      # Now, variables can be accessed normally, and have values assigned to them.
      print      (v.eval())
      print      (w.eval())
  print('--------初始化后，变量的值保留在同一会话中（不过，当您启动新会话时，需要重新初始化它们）：----------')
  with g.as_default():
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # These three prints will print the same value.
      print      (w.eval())
      print      (w.eval())
      print      (w.eval())
print('---更改变量的值，使用 assign 指令。创建 assign 指令不会起作用。须运行赋值指令才能更新变量值：---')
with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # This should print the variable's initial value.
    print (v.eval())
    assignment = tf.assign(v, [7])
    # The variable has not been changed yet!
    print (v.eval())
    # Execute the assignment op.
    sess.run(assignment)
    # Now the variable is updated.
    print (v.eval())
print('----------# 练习 2：模拟投掷两个骰子 10 次。-------------------------------------')
# 创建一个骰子模拟，在模拟中生成一个 10x3 二维张量，其中：#
# 列 1 和 2 均存储一个骰子的一次投掷值。# 列 3 存储同一行中列 1 和 2 的值的总和。
# 例如，第一行中可能会包含以下值：#
# 列 1 存储 4 # 列 2 存储 3 # 列 3 存储 7
with tf.Graph().as_default(), tf.Session() as sess:
  # Task 2: Simulate 10 throws of two dice. Store the results  # in a 10x3 matrix.
  # We're going to place dice throws inside two separate  # 10x1 matrices. We could have placed dice throws inside
  # a single 10x2 matrix, but adding different columns of  # the same matrix is tricky. We also could have placed
  # dice throws inside two 1-D tensors (vectors); doing so  # would require transposing the result.
  dice1 = tf.Variable(tf.random_uniform([10, 1],minval=1, maxval=7,dtype=tf.int32))
  dice2 = tf.Variable(tf.random_uniform([10, 1],  minval=1, maxval=7,dtype=tf.int32))
  # We may add dice1 and dice2 since they share the same shape   and size.
  dice_sum = tf.add(dice1, dice2)
  # We've got three separate 10x1 matrices. To produce a single 10x3 matrix, we'll concatenate them along dimension 1.
  resulting_matrix = tf.concat( values=[dice1, dice2, dice_sum], axis=1)
  # The variables haven't been initialized within the graph yet, so let's remedy that.
  sess.run(tf.global_variables_initializer())
  print(resulting_matrix.eval())