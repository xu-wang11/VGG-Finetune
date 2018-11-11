#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/11/11
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import math


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name, conv_shape, bias_shape):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        conv = tf.nn.conv2d(bottom, tf.Variable(tf.truncated_normal(conv_shape, stddev=0.1)), [1, 1, 1, 1], padding='SAME')

        bias = tf.nn.bias_add(conv, tf.Variable(tf.constant(0.1, shape=bias_shape)))

        relu = tf.nn.relu(bias)

        return relu


def fc_layer(input, name, output):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = input.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(input, [-1, dim])
        initial = tf.truncated_normal_initializer(0, 1/math.sqrt(dim))
        weights = tf.get_variable('weights', (dim, output), tf.float32, initializer=initial)
        biases = tf.get_variable('biases', (output,), tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def build_model(train_data, label_data, val_data, val_label):
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='X')
    y = tf.placeholder(tf.float32, shape=[None, 6], name='Y')
    conv1 = conv_layer(x, 'conv1', (3, 3, 1, 64), (64,))
    pool1 = avg_pool(conv1, 'pool1')
    conv2 = conv_layer(pool1, 'conv2', (3, 3, 64, 64), (64,))
    pool2 = avg_pool(conv2, 'pool2')

    # conv3 = conv_layer(pool2, 'conv3', (3, 3, 32, 16), (16,))
    # pool3 = avg_pool(conv3, 'pool3')

    conv4 = conv_layer(pool2, 'conv4', (3, 3, 64, 16), (16,))
    pool4 = avg_pool(conv4, 'pool4')
    fc1 = fc_layer(pool4, 'fc1', 512)
    fc1_dropout = tf.nn.dropout(tf.nn.relu(fc1), 0.5)
    fc2 = fc_layer(fc1_dropout, 'fc2', 6)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc2)
    op_loss = tf.reduce_mean(entropy, name='loss')
    op_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(op_loss)

    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    n_epochs = 300
    batch_size = 300
    num_examples = train_data.shape[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            true_num = 0
            sample_num = 0
            for iteration in range(num_examples // batch_size):
                # this cycle is for dividing step by step the heavy work of each neuron
                X_batch = train_data[iteration * batch_size:iteration * batch_size + batch_size, :, :]
                y_batch = label_data[iteration * batch_size:iteration * batch_size + batch_size]
                sess.run(op_opt, feed_dict={x: X_batch, y: y_batch})
            for iteration in range(test_data.shape[0] // batch_size):
                X_batch = test_data[iteration * batch_size:iteration * batch_size + batch_size, :, :]
                y_batch = test_label[iteration * batch_size:iteration * batch_size + batch_size]

                n = sess.run(correct_prediction, feed_dict={x: X_batch, y: y_batch})
                true_num += np.sum(n)
                sample_num += X_batch.shape[0]
            print("Validating data: " + str(true_num * 1.0 / sample_num))


if __name__ == '__main__':
    input = np.load('../data/RML/rml_audio_feature.npy')
    output = np.load('../data/RML/rml_label.npy')
    min_val = np.min(input)
    max_val = np.max(input)
    input = (input - min_val) / (max_val - min_val)
    # ind = [i for i in range(input.shape[0])]
    data_set = shuffle(input, output)
    train_len = int(data_set[0].shape[0] * 0.7)
    train_data = data_set[0][0:train_len]
    train_label = data_set[1][0:train_len]
    test_data = data_set[0][train_len:]
    test_label = data_set[1][train_len:]
    build_model(train_data, train_label, test_data, test_label)