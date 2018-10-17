#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/9/27
import tensorflow as tf
import time
import numpy as np
import pickle
import os
import pandas as pd


class VGGNet:
    def __init__(self, model_path='Weights_imageNet'):
        self.batch_size = 64
        self.cpu_cores = 8
        self.model_path = model_path
        self.weight_dict, self.bias_dict = pickle.load(open(model_path, 'rb'))
        print("loading weight matrix")
        self.skip_step = 100
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.lr = 0.0001

        self.op_opt = None
        self.op_loss = None
        self.op_summary = None

    def inference(self, x):
        output = self.construct_conv_layers(x)
        logits = self.construct_fc_layers(output)
        return logits

    def construct_conv_layers(self, input):
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

    def construct_fc_layers(self, input):
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

    def get_conv_filter(self, name):
        return tf.get_variable(name='weights', initializer=self.weight_dict[name])

    def get_bias(self, name):
        return tf.get_variable(name='biases', initializer=self.bias_dict[name])

    def get_fc_weight(self, name):
        return tf.get_variable(name="weights", initializer=self.weight_dict[name])

    @staticmethod
    def parse_image(filename, label):
        averageImg_BGR_imageNet = tf.constant(
            np.expand_dims(np.expand_dims(np.array([103.939, 116.779, 123.68], dtype=np.float32), axis=0), axis=0))
        image_string = tf.read_file(filename)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)
        h = tf.cast(tf.shape(image_decoded)[0], dtype=tf.float32)
        w = tf.cast(tf.shape(image_decoded)[1], dtype=tf.float32)

        def f1(): return tf.cast(256 / w * h, dtype=tf.int32)

        def f2(): return tf.cast(256 / h * w, dtype=tf.int32)

        def f3(): return tf.constant(256)

        h_r = tf.case({h <= w: f3, h > w: f1}, exclusive=True)
        w_r = tf.case({w <= h: f3, w > h: f2}, exclusive=True)
        image_resized = tf.image.resize_images(image_decoded, [h_r, w_r])
        image_cropped = tf.image.resize_image_with_crop_or_pad(image_resized, 224, 224)
        image_bgr = tf.reverse(image_cropped, axis=[-1])
        image_nml = image_bgr - averageImg_BGR_imageNet
        label = tf.one_hot(indices=label, depth=1000)
        return image_nml, label

    @staticmethod
    def parse_image_train(file_name, label):
        averageImg_BGR_imageNet = tf.constant(
            np.expand_dims(np.expand_dims(np.array([103.939, 116.779, 123.68], dtype=np.float32), axis=0), axis=0))
        image_string = tf.read_file(file_name)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)
        h = tf.cast(tf.shape(image_decoded)[0], dtype=tf.float32)
        w = tf.cast(tf.shape(image_decoded)[1], dtype=tf.float32)

        def f1(): return tf.cast(256 / w * h, dtype=tf.int32)

        def f2(): return tf.cast(256 / h * w, dtype=tf.int32)

        def f3(): return tf.constant(256)

        h_r = tf.case({h <= w: f3, h > w: f1}, exclusive=True)
        w_r = tf.case({w <= h: f3, w > h: f2}, exclusive=True)
        image_resized = tf.image.resize_images(image_decoded, [h_r, w_r])
        image_bgr = tf.reverse(image_resized, axis=[-1])
        image_nml = image_bgr - averageImg_BGR_imageNet
        image_cropped = tf.image.random_flip_left_right(tf.random_crop(image_nml, [224, 224, 3]))
        label = tf.one_hot(indices=label, depth=1000)
        return image_cropped, label

    # load dataset
    def load_dataset(self, imgs_path):
        # % load data
        img_dir = imgs_path
        file_paths = np.array([os.path.join(img_dir,  x) for x in sorted(os.listdir(img_dir))])

        labels = np.array(pd.read_csv('ILSVRC_labels.txt', delim_whitespace=True, header=None).values[:, 1], dtype=np.int32)

        train_index = np.ones(labels.shape, dtype=np.bool)
        val_index = np.zeros(labels.shape, dtype=np.bool)

        for i in range(1000):
            class_index = np.argwhere(labels == i)
            #        rand_index=np.random.choice(50,10, replace=False)
            rand_index = [25, 0, 24, 8, 37, 19, 3, 14, 15, 38]
            train_index[class_index[rand_index]] = False
            val_index[class_index[rand_index]] = True

        file_paths_train = file_paths[train_index]
        labels_train = labels[train_index]
        # nb_exp_train = len(labels_train)

        file_paths_val = file_paths[val_index]
        labels_val = labels[val_index]
        # self.n_test = len(labels_val)

        file_paths_train = tf.constant(file_paths_train)
        labels_train = tf.constant(labels_train)

        file_paths_val = tf.constant(file_paths_val)
        labels_val = tf.constant(labels_val)

        # %% construct input pipeline

        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=100000)
        dataset_train = dataset_train.repeat(count=4)
        dataset_train = dataset_train.map(map_func=self.parse_image_train, num_parallel_calls=self.cpu_cores)
        dataset_train = dataset_train.batch(self.batch_size)
        dataset_train = dataset_train.prefetch(buffer_size=1)

        dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
        dataset_val = dataset_val.map(map_func=self.parse_image, num_parallel_calls=self.cpu_cores)
        dataset_val = dataset_val.batch(self.batch_size)
        dataset_val = dataset_val.prefetch(buffer_size=1)

        vgg_iter = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        x, y = vgg_iter.get_next()

        train_init = vgg_iter.make_initializer(dataset_train)  # initializer for train_data
        test_init = vgg_iter.make_initializer(dataset_val)
        # return dataset_train, dataset_val, nb_exp_train, np_exp_val
        return train_init, test_init, x, y

    def loss(self, labels, logits):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.op_loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        var_list = []
        self.op_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=var_list,
                                                                                        global_step=self.gstep)

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.Y, 1))
            # self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            self.accuracy = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(predictions=preds, targets=tf.argmax(self.Y, axis=1), k=5),
                        dtype=tf.int32))

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.op_summary = tf.summary.merge_all()

    def build(self, x, y):
        '''
        Build the computation graph
        '''

        logits = self.inference(x)
        self.loss(y, logits)
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
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

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, train_init, test_init, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        writer = tf.summary.FileWriter('graphs/convnet', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, train_init, writer, epoch, step)
                self.eval_once(sess, test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    vgg = VGGNet()
    train_init, test_init, x, y = vgg.load_dataset(imgs_path='/srv/node/sdc1/image_data/img_val')
    vgg.build(x, y)
    vgg.train(train_init, test_init, n_epochs=1)

