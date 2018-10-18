#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
import pickle
import tensorflow as tf
from abc import abstractmethod

class VGGBase:
    """
    VGGBase is the class for inference. The training and evaluation method should be implemented in subclass.
    """

    def __init__(self):
        self.weight_dict = None
        self.bias_dict = None
        self.global_step = None

    def load_model(self, model_path):
        print("loading model from {0}".format(model_path))
        self.weight_dict, self.bias_dict = pickle.load(open(model_path, 'rb'))

    def save_model(self, save_path):
        pass

    def inference(self, x):
        print("start to build model...")
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        output = self.conv_layers(x)
        logits = self.fc_layers(output)
        return logits

    def conv_layers(self, input):
        # model definition
        conv1_1 = self.conv_layer(input, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')
        return pool5

    def fc_layers(self, input):
        fc6 = self.fc_layer(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)
        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)
        logits = self.fc_layer(relu7, "fc8")
        return logits

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def fc_layer_like(self, input, name):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])
            initial = tf.truncated_normal_initializer(0, 0.1)
            weights = tf.get_variable('weights', self.weight_dict[name].shape, tf.float32, initializer=initial)
            biases = tf.get_variable('biases', self.bias_dict[name].shape, tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    @staticmethod
    def output_layer(bottom, name, output_dim):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            initial = tf.truncated_normal_initializer(0, 0.1)
            weights = tf.get_variable('weights', (dim, output_dim), tf.float32, initializer=initial)
            biases = tf.get_variable('biases', (output_dim,), tf.float32, initializer=tf.constant_initializer(0.1))

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.get_variable(name='weights', initializer=self.weight_dict[name])

    def get_bias(self, name):
        return tf.get_variable(name='biases', initializer=self.bias_dict[name])

    def get_fc_weight(self, name):
        return tf.get_variable(name="weights", initializer=self.weight_dict[name])

    def build(self, x, y):
        logits = self.inference(x)
        self.loss(y, logits)
        self.optimize()
        self.prediction(y, logits)
        self.summary()

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def prediction(self, labels, logits):
        pass

    @abstractmethod
    def evaluation(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def summary(self):
        pass

