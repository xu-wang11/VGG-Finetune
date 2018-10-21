#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/9/27
import tensorflow as tf
import time
from vgg_base import VGGBase
import numpy as np
import utils


class VGGNet(VGGBase):
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

    def loss(self, labels, logits):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.op_loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        var_list = self.trainable_variables()
        self.op_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.op_loss, var_list=var_list,
                                                                                        global_step=self.global_step)

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        return var_list

    def prediction(self, labels, logits):
        with tf.name_scope('predict'):
            print(labels.shape)
            print(logits.shape)
            predictions = tf.nn.softmax(logits)
            self.preds = tf.cast(tf.nn.in_top_k(predictions=predictions, targets=tf.argmax(labels, axis=1), k=5),
                                      dtype=tf.int32)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(predictions=predictions, targets=tf.argmax(labels, axis=1), k=5),
                        dtype=tf.int32))

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.op_loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.op_loss)
            self.op_summary = tf.summary.merge_all()

    def build(self, x, y):
        logits = self.inference(x)
        self.loss(y, logits)
        self.optimize()
        self.prediction(y, logits)
        self.summary()

    def train_one_epoch(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.op_opt, self.op_loss, self.op_summary])
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

    def evaluation(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_samples = 0
        try:
            while True:
                batch_prediction, summaries = sess.run([self.preds, self.op_summary])
                batch_prediction = np.array(batch_prediction)
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += batch_prediction.sum()
                total_samples += batch_prediction.shape[1]
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init, test_init, n_epochs):
        writer = tf.summary.FileWriter('graphs/vgg_net', tf.get_default_graph())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            self.save_model(sess, 'vgg_net_before_train.data')
            step = self.global_step.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, train_init, writer, epoch, step)
                self.evaluation(sess, test_init, writer, epoch, step)
            self.save_model(sess, 'vgg_net_final_train.data')
        writer.close()


if __name__ == '__main__':
    vgg = VGGNet()
    train_set, val_set = utils.load_image_net_dataset(imgs_path='/srv/node/sdc1/image_data/img_val',
                                                               label_path='ILSVRC_labels.txt',
                                                               cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)
    train_init, test_init, x, y = utils.dataset_iterator(train_set, val_set)

    vgg.load_model(model_path='Weights_imageNet')
    vgg.build(x, y)
    vgg.train(train_init, test_init, n_epochs=1)

