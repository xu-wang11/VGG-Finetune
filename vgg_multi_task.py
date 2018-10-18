#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
from vgg_face import VGGFace
import tensorflow as tf
import time
import numpy as np

class VggMultiTask(VGGFace):

    def __init__(self):
        super().__init__()
        self.op_loss_celeba = None
        self.op_loss_imagenet = None
        self.op_opt_celeba = None
        self.op_opt_imagenet = None
        self.prediction_imagenet = None
        self.prediction_celeba = None
        self.accuracy_imagenet = None
        self.accuracy_celeba = None

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
            weights = tf.get_variable('weights', self.weight_dict[name].shape, tf.float32, initializer=initial)
            biases = tf.get_variable('biases', self.bias_dict[name].shape, tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        return var_list

    def loss_celeba(self, labels, logits):
        logits = tf.nn.sigmoid(logits)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        self.op_loss_celeba = tf.reduce_mean(loss, name='loss')

    def loss_imagenet(self, labels, logits):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.op_loss = tf.reduce_mean(entropy, name='loss')

    def optimize_celeba(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        var_list = self.trainable_variables()
        self.op_opt_celeba = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(
            self.op_loss_celeba, var_list=var_list, global_step=self.gstep)

    def optimize_imagenet(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        var_list = self.trainable_variables()
        self.op_opt_imagenet = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(
            self.op_loss_image_net, var_list=var_list, global_step=self.gstep)

    def eval_imagenet(self, labels, logits):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
            # self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            self.prediction_imagenet = tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels, axis=1), k=5),
                                      dtype=tf.int32)
            self.accuracy_imagenet = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels, axis=1), k=5),
                        dtype=tf.int32))

    def eval_celeba(self, labels, logits):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.sigmoid(logits)
            self.prediction_celeba = tf.reduce_sum(tf.cast(tf.equal(labels, tf.round(preds)), tf.float32), axis=1) / 40

            self.accuracy_celeba = tf.reduce_sum(self.prediction)

    def build(self, x, y1, y2):
        '''
        Build the computation graph
        '''

        logits = self.inference(x)
        self.loss_imagenet(y1, logits[0])
        self.loss_celeba(y2, logits[1])
        self.optimize_imagenet()
        self.optimize_celeba()
        self.eval_imagenet(y1, logits[0])
        self.eval_celeba(y2, logits[1])
        self.summary()

    def train_one_epoch_imagenet(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.op_opt_imagenet, self.op_loss_imagenet, self.op_summary])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once_imagenet(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_samples = 0
        try:
            while True:
                prediction_batch, summaries = sess.run([self.prediction_imagenet, self.op_summary])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += np.sum(prediction_batch)
                total_samples += prediction_batch.shape[0]
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train_one_epoch_celeba(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.op_opt_celeba, self.op_loss_celeba, self.op_summary])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once_celeba(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_samples = 0
        try:
            while True:
                prediction_batch, summaries = sess.run([self.prediction_celeba, self.op_summary])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += prediction_batch.sum()
                total_samples += prediction_batch.shape[0]
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init_imagenet, test_init_imagenet, train_init_celeba, test_init_celeba, n_epochs):
        writer = tf.summary.FileWriter('graphs/convnet', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch_imagenet(sess, train_init_imagenet, writer, epoch, step)
                self.eval_once_imagenet(sess, test_init_imagenet, writer, epoch, step)
                step = self.train_one_epoch_celeba(sess, train_init_celeba, writer, epoch, step)
                self.eval_once_celeba(sess, test_init_celeba, writer, epoch, step)
        writer.close()

    def run(self):
        train_init_imagenet, test_init_imagenet, x1, y1 = self.load_dataset(imgs_path='/srv/node/sdc1/image_data/img_val')
        train_init_celeba, test_init_celeba, x2, y2 = self.load_face_dataset(
            imgs_path='/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba')

        self.build(x1, y1, y2)
        self.train(train_init_imagenet, test_init_imagenet, train_init_celeba, test_init_celeba, n_epochs=20)


if __name__ == '__main__':
    vgg = VggMultiTask()
    vgg.run()





