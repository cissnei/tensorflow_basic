__author__ = 'cissnei'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using Sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1- hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy Computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
W_val = []
cost_val = []
hy_val = []
s_val= []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X:x_data, Y:y_data}
    for step in range(12001):
        sess.run(train, feed_dict=feed)
        s_val.append(step)
        cost_val.append(sess.run(cost, feed_dict=feed))
        if step % 300 ==3:
            print(step, sess.run(cost, feed_dict=feed))

# Accuracy Report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    # print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    plt.title('cost reduction')
    plt.plot(s_val, cost_val, 'r--', label='cost')
    plt.grid(True)
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('costs')
    plt.show()