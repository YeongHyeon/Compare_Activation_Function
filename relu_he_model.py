import tensorflow as tf
import numpy as np

def he_initializer(input_dim):
    return np.sqrt(2)/np.sqrt(input_dim)

def weight_variable(shape, stdev=0.1):
    initial = tf.truncated_normal(shape, stddev=stdev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def model(x=None, y_=None, keep_prob=None):
    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = weight_variable([5, 5, 1, 2]) * he_initializer(1)
    b_conv1 = bias_variable([2])

    W_conv2 = weight_variable([5, 5, 2, 4]) * he_initializer(2)
    b_conv2 = bias_variable([4])

    W_conv3 = weight_variable([5, 5, 4, 8]) * he_initializer(4)
    b_conv3 = bias_variable([8])

    W_conv4 = weight_variable([5, 5, 8, 16]) * he_initializer(8)
    b_conv4 = bias_variable([16])

    W_conv5 = weight_variable([5, 5, 16, 32]) * he_initializer(16)
    b_conv5 = bias_variable([32])

    W_conv6 = weight_variable([5, 5, 32, 64]) * he_initializer(32)
    b_conv6 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_pool1 = max_pool_2x2(h_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_pool1, W_conv4) + b_conv4)
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
    h_pool2 = max_pool_2x2(h_conv6)

    W_fc1 = weight_variable([7 * 7 * 64, 1024]) * he_initializer(1)
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10]) * he_initializer(1)
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy, cross_entropy
