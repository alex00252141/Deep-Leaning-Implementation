# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:47:45 2018

@author: DART_HSU
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

# shape (28, 28, 1)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第 1層卷積層 -> shape (28, 28, 1)
l1_conv = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)

# 第 1層池化層 -> shape (14, 14, 1)
l2_pool = tf.layers.max_pooling2d(l1_conv, pool_size=2, strides=2, padding='same')

# 第 2層卷積層 -> shape (14, 14, 1)
l3_conv = tf.layers.conv2d(inputs=l2_pool, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)

# 第 2層池化層 -> shape (7, 7, 1)
l4_pool = tf.layers.max_pooling2d(l3_conv, pool_size=2, strides=2, padding='same')

# 第 1層全連結層
flat1 = tf.reshape(l4_pool, [-1, 7*7*64])
# 輸出
fc_cnn_10 = tf.layers.dense(flat1, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fc_cnn_10)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(fc_cnn_10, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iteration: ' +str(epoch) + ', Accuracy: ' + str(acc))
        
    
        


