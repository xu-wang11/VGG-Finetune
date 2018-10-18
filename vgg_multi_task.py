#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
from vggnet import VGGNet
import tensorflow as tf

class VggMultiTask(VGGNet):

    def __init__(self):
        super().__init__()

    # override fc layer for CelebA dataset
    def construct_fc_layers(self, input):
        fc6 = self.fc_layer_like(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer_like(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        tasks_imagenet = self.fc_layer_like(relu7, "fc8")
        tasks_celeba = self.celeba_output_layer(relu7, "fc9")

        return [tasks_imagenet, tasks_celeba]

    def fc_layer_like(self, input, name):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])
            initial = tf.truncated_normal_initializer(0, 0.1)
            weights = tf.get_variable('weights', self.weight_dict[name], tf.float32, initializer=initial)
            biases = tf.get_variable('biases', self.bias_dict[name].shape, tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def celeba_output_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            initial = tf.truncated_normal_initializer(0, 0.1)
            weights = tf.get_variable('weights', (dim, 40), tf.float32, initializer=initial)
            biases = tf.get_variable('biases', (40,), tf.float32, initializer=tf.constant_initializer(0.1))

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        return var_list

    def loss_celeba(self, labels, logits):
        logits = tf.nn.sigmoid(logits)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        self.op_loss = tf.reduce_mean(loss, name='loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        var_list = self.trainable_variables()
        self.op_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.op_loss, var_list=var_list,
                                                                                        global_step=self.gstep)

