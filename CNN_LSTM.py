import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size


# 定義 CNN layer層
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷積層元件
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化層元件
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第 1層卷積層 
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第 1層池化層 
h_pool1 = max_pool_2x2(h_conv1)

# 第 2層卷積層
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第 2層池化層 
h_pool2 = max_pool_2x2(h_conv2)

# 第 1層全連結層
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop層
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第 2層全連結層
W_fc2 = weight_variable([1024, 784])
b_fc2 = bias_variable([784])
fc_cnn_10 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Training Parameters
n_inputs = 28
max_time = 28
lstm_size = 100
n_classes = 10

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
    prediction = RNN(fc_cnn_10, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
            
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1})
        print('Iteration: ' +str(epoch) + ', Accuracy: ' + str(acc))