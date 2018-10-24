#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
from vgg_base import VGGBase
import tensorflow as tf
import time
import numpy as np
import utils


class VggMultiTask(VGGBase):

    def __init__(self, is_from_vgg_weight=True):
        super().__init__()
        self.batch_size = 64
        self.cpu_cores = 8
        self.skip_step = 100
        self.lr = 0.0001
        self.is_from_vgg_weight = is_from_vgg_weight

        self.op_opt = None
        self.op_loss = None
        self.op_summary = None
        self.accuracy = None
        # self.preds = None
        self.pred_imagenet = None
        self.pred_celeba = None

    def inference(self, x):
        print("start to build model...")
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        output = self.conv_layers(x[0])
        output1 = self.conv_layers(x[1])
        logits = self.fc_layers([output, output1])
        return logits

    def fc_layers(self, input):
        fc6_0 = self.fc_layer(input[0], "fc6")

        relu6_0 = tf.nn.relu(fc6_0)

        fc7_0 = self.fc_layer(relu6_0, "fc7")
        relu7_0 = tf.nn.relu(fc7_0)

        fc6_1 = self.fc_layer(input[1], "fc6")

        relu6_1 = tf.nn.relu(fc6_1)

        fc7_1 = self.fc_layer(relu6_1, "fc7")
        relu7_1 = tf.nn.relu(fc7_1)

        tasks_imagenet = self.fc_layer(relu7_0, "fc8")
        if self.is_from_vgg_weight:
            tasks_celeba = self.output_layer(relu7_1, "fc9", 40)
        else:
            tasks_celeba = self.fc_layer(relu7_1, "fc9")

        return [tasks_imagenet, tasks_celeba]

    def trainable_variables(self):
        # var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        return tf.trainable_variables()

    def loss(self, labels, logits):

        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels[0], logits=logits[0])
        op_loss_imagenet = tf.reduce_mean(entropy, name='loss')

        loss = tf.losses.mean_squared_error(labels=labels[1], predictions=tf.nn.sigmoid(logits[1]))
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
            # print(labels[0].shape)
            # print(logits[0].shape)

            prediction_imagenet = tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels[0], axis=1), k=5),
                                      dtype=tf.int32)
            # print(prediction_imagenet.shape)
            accuracy_imagenet = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(labels[0], axis=1), k=5),
                        dtype=tf.int32))

            preds = tf.nn.sigmoid(logits[1])
            prediction_celeba = tf.reduce_sum(tf.cast(tf.equal(labels[1], tf.round(preds)), tf.float32), axis=1) / 40

            accuracy_celeba = tf.reduce_sum(prediction_celeba)

            self.pred_imagenet = prediction_imagenet
            self.pred_celeba = prediction_celeba
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
                _, l = sess.run([self.op_opt[0], self.op_loss[0]])

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
        # print("hell")
        try:
            while True:
                prediction_batch = sess.run([self.pred_imagenet])
                prediction_batch = np.array(prediction_batch)
                total_correct_preds += np.sum(prediction_batch)
                total_samples += prediction_batch.shape[1]
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
                _, l = sess.run([self.op_opt[1], self.op_loss[1]])

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
                prediction_batch = sess.run([self.pred_celeba])
                prediction_batch = np.array(prediction_batch)
                total_correct_preds += prediction_batch.sum()
                total_samples += prediction_batch.shape[1]
                # print(prediction_batch.shape)
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / total_samples))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init_imagenet, test_init_imagenet, train_init_celeba, test_init_celeba, n_epochs):
        writer = tf.summary.FileWriter('graphs/multitask', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.save_model(sess, 'vgg_multi_before_train.data')
            step = self.global_step.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch_celeba(sess, train_init_celeba, writer, epoch, step)
                self.evaluation_imagenet(sess, test_init_imagenet, writer, epoch, step)
                self.evaluation_celeba(sess, test_init_celeba, writer, epoch, step)

                step = self.train_one_epoch_imagenet(sess, train_init_imagenet, writer, epoch, step)
                self.evaluation_imagenet(sess, test_init_imagenet, writer, epoch, step)
                self.evaluation_celeba(sess, test_init_celeba, writer, epoch, step)

            self.save_model(sess, 'vgg_multi_after_train.data')
        writer.close()

    def evaluation(self, test_init_imagenet, test_init_celeba):
        writer = tf.summary.FileWriter('graphs/multitask', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.save_model(sess, 'vgg_multi_before_train.data')
            step = self.global_step.eval()
            self.evaluation_imagenet(sess, test_init_imagenet, writer, 0, step)
            self.evaluation_celeba(sess, test_init_celeba, writer, 0, step)
        writer.close()


if __name__ == '__main__':
    vgg = VggMultiTask(is_from_vgg_weight=False)
    train_set_image_net, test_set_image_net = utils.load_image_net_dataset(
        imgs_path='/srv/node/sdc1/image_data/img_val', label_path='ILSVRC_labels.txt',
        cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)

    train_set_celeba, test_set_celeba = utils.load_face_dataset(
        imgs_path='/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba',
        attr_file='/srv/node/sdc1/image_data/CelebA/Anno/list_attr_celeba.txt',
        partition_file='/srv/node/sdc1/image_data/CelebA/Eval/list_eval_partition.txt',
        cpu_cores=vgg.cpu_cores, batch_size=vgg.batch_size)

    train_init_image_net, test_init_image_net, x_image_net, y_image_net = utils.dataset_iterator(train_set_image_net, test_set_image_net)
    train_init_celeba, test_init_celeba, x_celeba, y_celeba = utils.dataset_iterator(train_set_celeba, test_set_celeba)

    # vgg.load_model('Weights_imageNet')
    # vgg.build([x_image_net, x_celeba], [y_image_net, y_celeba])
    # vgg.train(train_init_image_net, test_init_image_net, train_init_celeba, test_init_celeba, n_epochs=3)

    vgg.load_model('vgg_result/20181023/vgg_multi_after_train.data')
    vgg.build([x_image_net, x_celeba], [y_image_net, y_celeba])
    vgg.evaluation(test_init_image_net, test_init_celeba)
    # vgg.train(train_init_image_net, test_init_image_net, train_init_celeba, test_init_celeba, n_epochs=3)








