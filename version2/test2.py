import tensorflow as tf
import numpy as np


w=4
k=5
av=10
ae=5

a = tf.constant(1,shape=[1,w*k,av],dtype=tf.float32)
b = tf.constant(0,shape=[1,w*k*k,ae],dtype=tf.float32)
wa=tf.Variable(tf.truncated_normal([k,av,1],stddev=0.1))
wb=tf.Variable(tf.truncated_normal([k*k,ae,1],stddev=0.1))


ra=tf.nn.conv1d(a,wa,stride=k,padding='VALID')
#c=ra.shape
print(ra.shape)
rb=tf.nn.conv1d(b,wb,stride=k*k,padding='VALID')
print(rb.shape)
#d=rb.shape
wem=tf.Variable(tf.truncated_normal([]))
em = tf.nn.embedding_lookup([ra,rb],ids=[0,2])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    c=sess.run(em)
    print(c)
    #print(d)
