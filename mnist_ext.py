import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime

import relu_model
import sigmoid_model

def save_graph_as_image(train_list, test_list, ylabel=""):

    print(" Save "+ylabel+" graph in ./graph")

    x = np.arange(len(train_list))
    plt.clf()
    plt.plot(x, train_list, label="train "+ylabel)
    plt.plot(x, test_list, label="test "+ylabel, linestyle='--')
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, max([1, max(train_list), max(test_list)])*1.1)
    if(ylabel == "accuracy"):
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    #plt.show()

    if(not(os.path.exists("./graph"))):
        os.mkdir("./graph")
    else:
        pass
    now = datetime.now()

    # plt.savefig("./graph/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+ylabel+".png")
    plt.show()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

train_step_relu, accuracy_relu, cross_entropy_relu = relu_model.model(x=x, y_=y_, keep_prob=keep_prob)
train_step_sigmoid, accuracy_sigmoid, cross_entropy_sigmoid = sigmoid_model.model(x=x, y_=y_, keep_prob=keep_prob)

sess.run(tf.global_variables_initializer())

train_acc_list_relu = []
train_loss_list_relu = []
test_acc_list_relu = []
test_loss_list_relu = []

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        test_batch = mnist.test.next_batch(50)

        train_accuracy_relu = accuracy_relu.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_loss_relu = cross_entropy_relu.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})

        test_accuracy_relu = accuracy_relu.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        test_loss_relu = cross_entropy_relu.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})

        train_acc_list_relu.append(train_accuracy_relu)
        train_loss_list_relu.append(train_loss_relu)
        test_acc_list_relu.append(test_accuracy_relu)
        test_loss_list_relu.append(test_loss_relu)

        print("step %d, training accuracy \t%g \t%g"%(i, train_accuracy_relu, train_loss_relu))
    train_step_relu.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_graph_as_image(train_list=train_acc_list_relu, test_list=test_acc_list_relu, ylabel="Accuracy")
save_graph_as_image(train_list=train_loss_list_relu, test_list=test_loss_list_relu, ylabel="Loss")

print("test accuracy %g"%accuracy_relu.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
