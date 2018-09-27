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
    def __init__(self, model_path='Weights_imageNet', imgs_path=''):
        self.batch_size = 32
        self.cpu_cores = 8
        self.model_path = model_path
        self.imgs_path = imgs_path
        self.weight_dict, self.bias_dict = pickle.load(open(model_path, 'rb'))
        print("loading weight matrix")

    def build(self):
        start_time = time.time()
        train_dataset, val_dataset, num_train, num_val = self.load_imageNet()
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        X, Y = iter.get_next()

        # model definition
        self.conv1_1 = self.conv_layer(X, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.logits = tf.nn.softmax(self.fc8, name="prob")

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=self.logits)

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(self.loss, var_list=[tf.get_variable('fc8')])

        accuracy_sum = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(predictions=tf.argmax(self.logits, axis=1), targets=Y, k=5),
                dtype=tf.int32))
        print(("build model finished: %ds" % (time.time() - start_time)))

        with tf.Session() as sess:
            sess.run(iter.make_iterator(val_dataset))
            for i in range(1, 100):
                accu_sum = sess.run(accuracy_sum)
                print(accu_sum)

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

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.weight_dict[name], name="filter")

    def get_bias(self, name):
        return tf.constant(self.bias_dict[name], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.weight_dict[name], name="weights")

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
    def load_dataset(self, img_dir):
        # % load data
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
        nb_exp_train = len(labels_train)

        file_paths_val = file_paths[val_index]
        labels_val = labels[val_index]
        np_exp_val = len(labels_val)

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

        return dataset_train, dataset_val, nb_exp_train, np_exp_val


if __name__ == '__main__':
    vgg = VGGNet(imgs_path='/srv/node/sdc1/image_data/img_val')
    vgg.build()
