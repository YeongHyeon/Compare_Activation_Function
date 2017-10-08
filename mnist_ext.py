import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime

import relu_model
import sigmoid_model
import relu_he_model

def save_graph_as_image(train_list, test_list, ylabel="", label1="train", label2="test", cate="None"):

    print("Save "+ylabel+" graph in ./graph")

    x = np.arange(len(train_list))
    plt.clf()
    plt.plot(x, train_list, label=label1, linestyle='--')
    plt.plot(x, test_list, label=label2, linestyle='--')
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

    plt.savefig("./graph/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+cate+"_"+ylabel+".png")
    # plt.show()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

train_step_relu, accuracy_relu, cross_entropy_relu = relu_model.model(x=x, y_=y_, keep_prob=keep_prob)
train_step_sigmoid, accuracy_sigmoid, cross_entropy_sigmoid = sigmoid_model.model(x=x, y_=y_, keep_prob=keep_prob)
train_step_he, accuracy_he, cross_entropy_he = relu_he_model.model(x=x, y_=y_, keep_prob=keep_prob)

sess.run(tf.global_variables_initializer())

train_acc_list_relu = []
train_loss_list_relu = []
test_acc_list_relu = []
test_loss_list_relu = []

train_acc_list_sigmoid = []
train_loss_list_sigmoid = []
test_acc_list_sigmoid = []
test_loss_list_sigmoid = []

train_acc_list_he = []
train_loss_list_he = []
test_acc_list_he = []
test_loss_list_he = []

print("\nTraining")
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        test_batch = mnist.test.next_batch(50)

        train_accuracy_relu = accuracy_relu.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_loss_relu = np.nan_to_num(cross_entropy_relu.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0}))

        test_accuracy_relu = accuracy_relu.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        test_loss_relu = np.nan_to_num(cross_entropy_relu.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

        train_acc_list_relu.append(train_accuracy_relu)
        train_loss_list_relu.append(train_loss_relu)
        test_acc_list_relu.append(test_accuracy_relu)
        test_loss_list_relu.append(test_loss_relu)

        train_accuracy_sigmoid = accuracy_sigmoid.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_loss_sigmoid = np.nan_to_num(cross_entropy_sigmoid.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0}))

        test_accuracy_sigmoid = accuracy_sigmoid.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        test_loss_sigmoid = np.nan_to_num(cross_entropy_sigmoid.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

        train_acc_list_sigmoid.append(train_accuracy_sigmoid)
        train_loss_list_sigmoid.append(train_loss_sigmoid)
        test_acc_list_sigmoid.append(test_accuracy_sigmoid)
        test_loss_list_sigmoid.append(test_loss_sigmoid)

        train_accuracy_he = accuracy_he.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_loss_he = np.nan_to_num(cross_entropy_he.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0}))

        test_accuracy_he = accuracy_he.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        test_loss_he = np.nan_to_num(cross_entropy_he.eval(feed_dict={
            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

        train_acc_list_he.append(train_accuracy_he)
        train_loss_list_he.append(train_loss_he)
        test_acc_list_he.append(test_accuracy_he)
        test_loss_list_he.append(test_loss_he)

        print("step %d, training accuracy | %.4f %2.4f |%.4f %2.4f |%.4f %2.4f"%(i, train_accuracy_relu, train_loss_relu, train_accuracy_sigmoid, train_loss_sigmoid, train_accuracy_he, train_loss_he))
    train_step_relu.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_step_sigmoid.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_step_he.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_graph_as_image(train_list=train_acc_list_relu, test_list=test_acc_list_relu, ylabel="Accuracy", cate="ReLU")
save_graph_as_image(train_list=train_loss_list_relu, test_list=test_loss_list_relu, ylabel="Loss", cate="ReLU")

save_graph_as_image(train_list=train_acc_list_sigmoid, test_list=test_acc_list_sigmoid, ylabel="Accuracy", cate="Sigmoid")
save_graph_as_image(train_list=train_loss_list_sigmoid, test_list=test_loss_list_sigmoid, ylabel="Loss", cate="Sigmoid")

save_graph_as_image(train_list=train_acc_list_he, test_list=test_acc_list_he, ylabel="Accuracy", cate="He")
save_graph_as_image(train_list=train_loss_list_he, test_list=test_loss_list_he, ylabel="Loss", cate="He")

save_graph_as_image(train_list=train_acc_list_relu, test_list=train_acc_list_sigmoid, ylabel="Accuracy", label1="ReLU", label2="Sigmoid", cate="R_VS_S")
save_graph_as_image(train_list=train_loss_list_relu, test_list=train_loss_list_sigmoid, ylabel="Loss", label1="ReLU", label2="Sigmoid", cate="R_VS_S")

save_graph_as_image(train_list=train_acc_list_relu, test_list=train_acc_list_he, ylabel="Accuracy", label1="ReLU", label2="He", cate="R_VS_H")
save_graph_as_image(train_list=train_loss_list_relu, test_list=train_loss_list_he, ylabel="Loss", label1="ReLU", label2="He", cate="R_VS_H")
