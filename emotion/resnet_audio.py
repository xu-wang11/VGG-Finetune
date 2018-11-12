#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/11/12

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/9/27
import tensorflow as tf
import time
import numpy as np
import pickle
from sklearn.utils import shuffle
import os
#import pandas as pd

tf.reset_default_graph()


class ResNet:
    def __init__(self, model_path=[], dropout=False):
        tf.reset_default_graph()
        self.training = tf.placeholder(tf.bool, name='training')
        self.batch_size = 128
        self.cpu_cores = 8
        self.n_classes = 6
        self.model_path = model_path

        self.regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)
        self.n_group = 3
        self.n_blocks_per_group = 4
        self.width = 32

        self.dropout = dropout

        self.layer_input = dict()
        self.layer_output = dict()

        if not model_path:
            self.initial_weight = True
            self.weight_dict = self.construct_initial_weights()
        else:
            self.weight_dict = pickle.load(open(model_path, 'rb'))
            print("loading weight matrix")
            self.initial_weight = False

        self.skip_step = 100
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.lr = 0.1
        self.X = None
        self.Y = None


    def construct_initial_weights(self):
        weight_dict = dict()

        weight_dict['pre_conv'] = {
                'weights': np.random.normal(loc=0., scale=np.sqrt(1/(3*3*3)), size=[3,3,1,self.width]).astype(np.float32),
                'beta': np.zeros(self.width, dtype=np.float32),
                'mean': np.zeros(self.width, dtype=np.float32),
                'variance': np.ones(self.width, dtype=np.float32)}

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)
                if j == 0:
                    input_width=self.width*2**i
                else:
                    input_width=self.width*2**(i+1)

                weight_dict[block_name+'/conv_1'] = {
                        'weights': np.random.normal(loc=0., scale=np.sqrt(1/(3*3*input_width)), size=[3,3,input_width,self.width*2**(i+1)]).astype(np.float32),
                        'beta': np.zeros(self.width*2**(i+1), dtype=np.float32),
                        'mean': np.zeros(self.width*2**(i+1), dtype=np.float32),
                        'variance': np.ones(self.width*2**(i+1), dtype=np.float32)}

                weight_dict[block_name+'/conv_2'] = {
                        'weights': np.random.normal(loc=0., scale=np.sqrt(1/(3*3*self.width*2**(i+1))), size=[3,3,self.width*2**(i+1),self.width*2**(i+1)]).astype(np.float32),
                        'beta': np.zeros(self.width*2**(i+1), dtype=np.float32),
                        'mean': np.zeros(self.width*2**(i+1), dtype=np.float32),
                        'variance': np.ones(self.width*2**(i+1), dtype=np.float32)}

        weight_dict['end_bn'] = {
                'beta': np.zeros(self.width*2**3, dtype=np.float32),
                'mean': np.zeros(self.width*2**3, dtype=np.float32),
                'variance': np.ones(self.width*2**3, dtype=np.float32)}

        weight_dict['classifier'] = {
                'weights': np.random.normal(loc=0., scale=np.sqrt(1/256), size=[256,self.n_classes]).astype(np.float32),
                'biases': np.zeros(self.n_classes, dtype=np.float32)}

        return weight_dict

    def fetch_weight(self):
        weight_dict_tensor = dict()

        with tf.variable_scope('pre_conv',reuse=True):
            weight_dict_tensor['pre_conv'] = {
                    'weights': tf.get_variable('weights'),
                    'beta': tf.get_variable('batch_normalization/beta'),
                    'mean': tf.get_variable('batch_normalization/moving_mean'),
                    'variance': tf.get_variable('batch_normalization/moving_variance')}

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)

                with tf.variable_scope(block_name+'/conv_1',reuse=True):
                    weight_dict_tensor[block_name+'/conv_1'] = {
                            'weights': tf.get_variable('weights'),
                            'beta': tf.get_variable('batch_normalization/beta'),
                            'mean': tf.get_variable('batch_normalization/moving_mean'),
                            'variance': tf.get_variable('batch_normalization/moving_variance')}

                with tf.variable_scope(block_name+'/conv_2',reuse=True):
                    weight_dict_tensor[block_name+'/conv_2'] = {
                            'weights': tf.get_variable('weights'),
                            'beta': tf.get_variable('batch_normalization/beta'),
                            'mean': tf.get_variable('batch_normalization/moving_mean'),
                            'variance': tf.get_variable('batch_normalization/moving_variance')}

        with tf.variable_scope('end_bn',reuse=True):
            weight_dict_tensor['end_bn'] = {
                    'beta': tf.get_variable('batch_normalization/beta'),
                    'mean': tf.get_variable('batch_normalization/moving_mean'),
                    'variance': tf.get_variable('batch_normalization/moving_variance')}

        with tf.variable_scope('classifier',reuse=True):
            weight_dict_tensor['classifier'] = {
                    'weights': tf.get_variable('weights'),
                    'biases': tf.get_variable('biases')}

        return self.sess.run(weight_dict_tensor)

    def save_weight(self, save_path):
        self.weight_dict=self.fetch_weight()
        filehandler = open(save_path, 'wb')
        pickle.dump(self.weight_dict,filehandler)
        filehandler.close()

    def inference(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, 6], name='Y')

        # model definition
        self.layer_input['pre_conv'] = self.X
        y = self.conv_layer_basic(self.X, "pre_conv")
        self.layer_output['pre_conv'] = y

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)
                scale_down = j==0
                self.layer_input[block_name] = y
                y = self.res_block(y, name=block_name, scale_down=scale_down)
                self.layer_output[block_name] = y

        y = self.bn_layer(y, 'end_bn')
        y = tf.nn.relu(y)
        y = tf.reduce_mean(y,axis=[1,2])
        self. logits = self.fc_layer(y, 'classifier')

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer_basic(self, bottom, name, stride=1):
        with tf.variable_scope(name):
            filt, beta, mean, variance = self.get_conv_filter_bn()

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')

            if self.dropout:
                conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0],1,1,tf.shape(conv)[3]], training=self.training)

            bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training = self.training, beta_initializer=beta, scale=False, moving_mean_initializer=mean, moving_variance_initializer=variance, beta_regularizer=self.regularizer_conv)



            return bn

    def bn_layer(self, bottom, name):
        with tf.variable_scope(name):
            beta, mean, variance = self.get_bn_param()

            bn = tf.layers.batch_normalization(bottom, momentum=0.1, epsilon=1e-05, training = self.training, beta_initializer=beta, scale=False, moving_mean_initializer=mean, moving_variance_initializer=variance, beta_regularizer=self.regularizer_conv)

            return bn

    def res_block(self, bottom, name, scale_down=False):
        with tf.variable_scope(name):
            if scale_down:
                stride = 2
            else:
                stride = 1
            conv_1 = self.conv_layer_basic(bottom, 'conv_1', stride=stride)
            conv_1_relu = tf.nn.relu(conv_1)
            conv_2 = self.conv_layer_basic(conv_1_relu, 'conv_2')
            residual = bottom
            if scale_down:
                residual = tf.nn.avg_pool(residual, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                residual = tf.concat([residual,residual*0.],axis=-1)

            end_relu = tf.nn.relu(conv_2+residual)

            return end_relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_param()

            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

            return fc

    def get_conv_filter_bn(self):

        scope_name=tf.get_variable_scope().name

        filt = tf.get_variable(name="weights", initializer=self.weight_dict[scope_name]['weights'], regularizer=self.regularizer_conv)

        beta = tf.constant_initializer(self.weight_dict[scope_name]['beta'])
        mean = tf.constant_initializer(self.weight_dict[scope_name]['mean'])
        variance = tf.constant_initializer(self.weight_dict[scope_name]['variance'])

        return filt, beta, mean, variance

    def get_bn_param(self):
        scope_name=tf.get_variable_scope().name

        beta = tf.constant_initializer(self.weight_dict[scope_name]['beta'])
        mean = tf.constant_initializer(self.weight_dict[scope_name]['mean'])
        variance = tf.constant_initializer(self.weight_dict[scope_name]['variance'])

        return beta, mean, variance
#    def get_conv_filter(self, name):
#        return tf.get_variable(name="filter", initializer=self.weight_dict[name])

    def get_fc_param(self):
        scope_name=tf.get_variable_scope().name

        weights = tf.get_variable(name="weights", initializer=self.weight_dict[scope_name]['weights'], regularizer=self.regularizer_fc)
        biases = tf.get_variable(name="biases", initializer=self.weight_dict[scope_name]['biases'], regularizer=self.regularizer_fc)

        return weights, biases

    def get_bias(self, name):
        if name.startswith('fc'):
            return tf.get_variable(name='biases', initializer=self.bias_dict[name])
        else:
            return tf.constant(self.bias_dict[name], name="biases")

    def get_fc_weight(self, name):
        return tf.get_variable(name="weights", initializer=self.weight_dict[name])
        # return tf.constant(self.weight_dict[name], name="weights")

    def construct_loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits)
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(entropy, name='loss')+l2_loss

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt=tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=True)
            self.opt_op =self.opt.minimize(self.loss, global_step=self.gstep)

    def evaluate(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
#            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
#            self.accuracy = tf.reduce_sum(
#                tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=tf.argmax(self.Y, axis=1), k=1),dtype=tf.int32))

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        '''
        Build the computation graph
        '''

        self.inference()
        self.construct_loss()
        self.optimize()
        self.evaluate()
        self.summary()
        self.sess = tf.Session()
        self.initialize()

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def train_one_epoch(self, sess, train_data, train_label, writer, epoch, step):
#        start_time = time.time()

        total_loss = 0
        n_batches = 0
        time_last = time.time()

        try:
            for iteration in range(train_data.shape[0] // self.batch_size):
                # this cycle is for dividing step by step the heavy work of each neuron
                X_batch = train_data[iteration * self.batch_size:iteration * self.batch_size + self.batch_size, :, :]
                y_batch = train_label[iteration * self.batch_size:iteration * self.batch_size + self.batch_size]

                _, l, summaries = sess.run([self.opt_op, self.loss, self.summary_op],feed_dict={self.X: X_batch,
                                                                                                self.Y: y_batch,
                                                                                                'training:0': True})
                writer.add_summary(summaries, global_step=step)
#                if (step + 1) % self.skip_step == 0:
#                    print('Loss at step {0}: {1}'.format(step+1, l))
                step += 1
                total_loss += l
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
#        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
#        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, train_data, train_label, writer, epoch, step):
#        start_time = time.time()

        total_loss = 0
        total_correct_preds = 0
        n_batches = 0

        try:
            for iteration in range(train_data.shape[0] // self.batch_size):
                # this cycle is for dividing step by step the heavy work of each neuron
                X_batch = train_data[iteration * self.batch_size:iteration * self.batch_size + self.batch_size, :, :]
                y_batch = train_label[iteration * self.batch_size:iteration * self.batch_size + self.batch_size]
                loss_batch, accuracy_batch, summaries = sess.run([self.loss, self.accuracy, self.summary_op],
                                                                 feed_dict={self.X: X_batch, self.Y: y_batch,
                                                                            'training:0': False})
                writer.add_summary(summaries, global_step=step)
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                # print(accuracy_batch)
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}'.format(epoch+1, total_correct_preds / train_data.shape[0], total_loss / n_batches))
#        print('Val loss at epoch {:d}: {:f}'.format(epoch, total_loss / n_batches))
#        print('Accuracy at epoch {:d}: {:%} '.format(epoch, total_correct_preds / self.n_samples_val))
#        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs, train_data, train_label, test_data, test_label, lr=None):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        writer = tf.summary.FileWriter('graphs/convnet', tf.get_default_graph())
        if lr is not None:
            self.lr = lr
            self.optimize()

        self.sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.gstep.eval(session=self.sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(self.sess, train_data, train_label, writer, epoch, step)
            self.eval_once(self.sess, test_data, test_label, writer, epoch, step)
        writer.close()


    def test(self):
        self.sess.run(self.test_init)
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0

        try:
            while True:
                loss_batch, accuracy_batch, summaries = self.sess.run([self.loss, self.accuracy, self.summary_op],feed_dict={'training:0':False})
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('\nTesting, val_acc={:%}, val_loss={:f}'.format(total_correct_preds / self.n_samples_val, total_loss / n_batches))

#%%

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
    val_data = data_set[0][train_len:]
    val_label = data_set[1][train_len:]

    resnet = ResNet()
    resnet.build()

    resnet.train(80, train_data, train_label, val_data, val_label, lr=0.1)

    resnet.train(20, train_data, train_label, val_data, val_label, lr=0.01)

    resnet.train(20, train_data, train_label, val_data, val_label,  lr=0.001)
