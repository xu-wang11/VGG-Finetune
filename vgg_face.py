#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/17
import tensorflow as tf
from vgg_base import VGGBase
import time
import numpy as np
import utils


class VGGFace(VGGBase):

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

    # override fc layer for CelebA dataset
    def fc_layers(self, input):
        fc6 = self.fc_layer(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        logits = self.output_layer(relu7, "fc8", 40)

        return logits

    def loss(self, labels, logits):
        logits = tf.nn.sigmoid(logits)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        self.op_loss = tf.reduce_mean(loss, name='loss')

    def optimize(self):
        var_list = self.trainable_variables()
        self.op_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.op_loss, var_list=var_list,
                                                                                        global_step=self.global_step)

    def prediction(self, labels, logits):

        with tf.name_scope('predict'):
            preds = tf.nn.sigmoid(logits)
            self.preds = tf.reduce_sum(tf.cast(tf.equal(labels, tf.round(preds)), tf.float32), axis=1) / 40
            self.accuracy = tf.reduce_sum(self.preds)

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc8")]
        return tf.trainable_variables()

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.op_loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.op_loss)
            self.op_summary = tf.summary.merge_all()

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
                print(batch_prediction)
                exit(0)
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += batch_prediction.sum()
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init, test_init, n_epochs):
        writer = tf.summary.FileWriter('graphs/vgg_net', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.save_model(sess, 'vgg_face_after_train.data')
            step = self.global_step.eval()
            for epoch in range(n_epochs):
                self.evaluation(sess, test_init, writer, epoch, step)
                step = self.train_one_epoch(sess, train_init, writer, epoch, step)
                self.evaluation(sess, test_init, writer, epoch, step)
            self.save_model(sess, 'vgg_face_after_train.data')
        writer.close()


if __name__ == '__main__':
    vgg = VGGFace()
    train_set, val_set = utils.load_face_dataset(imgs_path='/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba',
                                                          attr_file='/srv/node/sdc1/image_data/CelebA/Anno/list_attr_celeba.txt',
                                                          partition_file='/srv/node/sdc1/image_data/CelebA/Eval/list_eval_partition.txt',
                                                          cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)
    train_init, test_init, x, y = utils.dataset_iterator(train_set, val_set)
    vgg.load_model(model_path='Weights_imageNet')
    vgg.build(x, y)
    vgg.train(train_init, test_init, n_epochs=20)


