__author__ = 'cissnei'

import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5],
          [1,2,5,6], [1,6,6,6], [1,7,7,7]]
# one-hot encoding
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = ext(Logits) / reduce_sum(exp(Logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

cost_val = []
step_val = []
op_val = []

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # print(sess.run(optimizer, feed_dict={X:x_data, Y:y_data}))
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        cost_val.append(sess.run(cost, feed_dict={X:x_data, Y:y_data}))
        step_val.append(step)
        if step % 200==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
    print(all, sess.run(tf.arg_max(all, 1)))

    plt.title('cost reduction')
    plt.plot(step_val, cost_val, 'b', label='cost')
    plt.grid(True)
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('costs')
    plt.show()
