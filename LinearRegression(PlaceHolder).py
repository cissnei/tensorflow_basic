__author__ = 'cissnei'

import tensorflow as tf

W = tf.Variable(tf.random_normal([1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Hypothesis XW+b
hypothesis = X * W + b

# Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line with new training data
for step in range(2501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: [1,3,5,7,9],
                                                    Y:[3.2,9.2,15.2,21.2,27.2]})
    if step % 20 ==0:
        print("step: ", step, "cost_val: ", cost_val,
              "Weight: ", W_val, "bias: ", b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5,5.5]}))