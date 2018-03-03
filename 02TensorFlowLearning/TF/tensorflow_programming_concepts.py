# 张量是任意维度的数组
# 低维张量：
# 标量是零维数组（零阶张量）。例如，\'Howdy\' 或 5
# 矢量是一维数组（一阶张量）。例如，[2, 3, 5, 7, 11] 或 [5]
# 矩阵是二维数组（二阶张量）。例如，[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]
# TensorFlow 指令:会创建、销毁和操控张量。
# TensorFlow 图（也称为计算图或数据流图）:是一种图数据结构。
# 图必须在 TensorFlow 会话中运行，会话存储了它所运行的图的状态.

import matplotlib.pyplot as plt # 数据集可视化。
import numpy as np              # 低级数字 Python 库。
import pandas as pd             # 较高级别的数字 Python 库。

import tensorflow as tf

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  sum = tf.add(x, y, name="x_y_sum")


  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
      print(sum.eval())

# Establish our graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of three operations.
    # (Creating a tensor is an operation.)
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")

    # Task 1: Define a third scalar integer constant z.
    z = tf.constant(4, name="z_const")
    # Task 2: Add z to `sum` to yield a new sum.
    new_sum = tf.add(sum, z, name="x_y_z_sum")

    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        # Task 3: Ensure the program yields the correct grand total.
        print(new_sum.eval())