__author__ = 'cissnei'

import tensorflow as tf

# X and Y data Definition

x_train = [1,2,4,6]
y_train = [2,8,16,32]

W = tf.Variable(tf.random_normal([1], name='weight'))
# tensor variable vector shape

b = tf.Variable(tf.random_normal([1], name='bias'))

# Hypothesis XW+b
hypothesis = x_train * W + b

# Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(3001):
    sess.run(train)
    if step % 40 ==0:
        print("step: ", step, "cost: ", sess.run(cost),
              "Weight: ", sess.run(W), "bias: ", sess.run(b))
