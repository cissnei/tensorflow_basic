__author__ = 'cissnei'

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

print("node1: ", node1) # it prints just structure of tensor

sess=tf.Session()
print(sess.run([node1, node2]))
print(sess.run(node3))
