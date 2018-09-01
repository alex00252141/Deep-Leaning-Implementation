"""
Author: CHIN JUNG, HSU

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
n_inputs = 28
max_time = 28
lstm_size = 100
n_classes = 10
batch_size = 50
n_batch = mnist.train.num_examples // batch_size 

# namespace
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

# Define weights
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1), name='w')
    with tf.name_scope('biases'):    
        biases = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
    
    def RNN(x, weights, biases):
        # transpose the inputs shape from
        inputs = tf.reshape(x, [-1, max_time, n_inputs])
        
        # cell
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        
        # basic LSTM Cell.
        outputs, final_states = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)
        
        # hidden layer for output as the final results
        results = tf.matmul(final_states[1], weights) + biases
     
        return results
    with tf.name_scope('RNN'):  
        pred = RNN(x, weights, biases)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # writer
    writer = tf.summary.FileWriter('logs/', sess.graph)
    
    # Run the initializer
    sess.run(init)
    
    for step in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            
        print(":" + str(step) + ", accuracy" +str(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))
        
        
writer.close()