__author__ = 'cissnei'
import tensorflow as tf

hello = tf.constant("hello, Tensorflows")
sess = tf.Session()

print(sess.run(hello))