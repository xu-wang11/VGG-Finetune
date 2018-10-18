#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
from vgg_base import VGGBase
import tensorflow as tf
import time
import numpy as np
import utils


class VggMultiTask(VGGBase):

    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.cpu_cores = 8
        self.skip_step = 100
        self.lr = 0.0001

        self.op_opt = None
        self.op_loss = None
        self.op_summary = None
        self.accuracy = None
        self.preds = None

    def fc_layers(self, input):
        fc6 = self.fc_layer_like(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer_like(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        tasks_imagenet = self.fc_layer_like(relu7, "fc8")
        tasks_celeba = self.output_layer(relu7, "fc9", 40)

        return [tasks_imagenet, tasks_celeba]

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        return var_list

    def loss(self, labels, logits):

        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels[0], logits=logits[0])
        op_loss_imagenet = tf.reduce_mean(entropy, name='loss')

        logits = tf.nn.sigmoid(logits[1])
        loss = tf.losses.mean_squared_error(labels=labels[1], predictions=logits)
        op_loss_celeba = tf.reduce_mean(loss, name='loss')

        self.op_loss = [op_loss_imagenet, op_loss_celeba]

    def optimize(self):

        var_list = self.trainable_variables()
        op_opt_imagenet = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(
            self.op_loss[0], var_list=var_list, global_step=self.global_step)
        op_opt_celeba = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(
            self.op_loss[1], var_list=var_list, global_step=self.global_step)
        self.op_opt = [op_opt_imagenet, op_opt_celeba]

    def prediction(self, labels, logits):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(logits[0])

            prediction_imagenet = tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels[0], axis=1), k=5),
                                      dtype=tf.int32)
            accuracy_imagenet = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels[0], axis=1), k=5),
                        dtype=tf.int32))

            preds = tf.nn.sigmoid(logits[1])
            prediction_celeba = tf.reduce_sum(tf.cast(tf.equal(labels[1], tf.round(preds)), tf.float32), axis=1) / 40

            accuracy_celeba = tf.reduce_sum(prediction_celeba)

            self.preds = [prediction_imagenet, prediction_celeba]
            self.accuracy = [accuracy_imagenet, accuracy_celeba]

    def summary(self):
        with tf.name_scope('summaries'):

            self.op_summary = tf.summary.merge_all()

    def train_one_epoch_imagenet(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.op_opt[0], self.op_loss[0], self.op_summary])
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

    def evaluation_imagenet(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_samples = 0
        try:
            while True:
                prediction_batch = sess.run([self.preds[0]])
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
                _, l, summaries = sess.run([self.op_opt[1], self.op_loss[1], self.op_summary])
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

    def evaluation_celeba(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_samples = 0
        try:
            while True:
                prediction_batch = sess.run([self.preds[1]])
                total_correct_preds += prediction_batch.sum()
                total_samples += prediction_batch.shape[0]
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init_imagenet, test_init_imagenet, train_init_celeba, test_init_celeba, n_epochs):
        writer = tf.summary.FileWriter('graphs/multitask', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = self.global_step.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch_imagenet(sess, train_init_imagenet, writer, epoch, step)
                self.evaluation_imagenet(sess, test_init_imagenet, writer, epoch, step)
                step = self.train_one_epoch_celeba(sess, train_init_celeba, writer, epoch, step)
                self.evaluation_celeba(sess, test_init_celeba, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    vgg = VggMultiTask()
    train_init_image_net, test_init_image_net, x_image_net, y_image_net = utils.load_image_net_dataset(
        imgs_path='/srv/node/sdc1/image_data/img_val', label_path='ILSVRC_labels.txt',
        cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)

    train_init_celeba, test_init_celeba, x_celeba, y_celeba = utils.load_face_dataset(
        imgs_path='/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba',
        attr_file='/srv/node/sdc1/image_data/CelebA/Anno/list_attr_celeba.txt',
        partition_file='/srv/node/sdc1/image_data/CelebA/Eval/list_eval_partition.txt',
        cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)

    vgg.load_model('Weights_imageNet')
    vgg.build(x_image_net, [y_image_net, y_celeba])
    vgg.train(train_init_image_net, test_init_image_net, train_init_celeba, test_init_celeba, n_epochs=20)








