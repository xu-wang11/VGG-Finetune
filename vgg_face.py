#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/17
from vggnet import VGGNet
import tensorflow as tf
import numpy as np
import pandas as pd
import os


class VGGFace(VGGNet):

    def __init__(self):
        pass

    # override fc layer for CelebA dataset
    def construct_fc_layers(self, input):
        fc6 = self.fc_layer(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        logits = self.celeba_output_layer(relu7, "fc8")

        return logits

    def celeba_output_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable(name, (dim, 10), tf.float32, initializer=tf.contrib.layers.xavier_initializer)
            biases = tf.get_variable(name, (10,), tf.float32, initializer=tf.zeros_initializer)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    @staticmethod
    def parse_image(filename, label):
        average_face = tf.constant(
            np.expand_dims(np.expand_dims(np.array([93.5940, 104.7624, 129.1863], dtype=np.float32), axis=0), axis=0))
        image_string = tf.read_file(filename)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)
        image_bgr = tf.reverse(image_resized, axis=[-1])
        image_nml = image_bgr - average_face
        return image_nml, label

    def trainable_variables(self):
        var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc8")]
        return var_list

    def load_dataset(self, imgs_path):
        img_dir = imgs_path
        file_paths = np.array([os.path.join(img_dir, x) for x in sorted(os.listdir(img_dir))])

        labels = np.array(pd.read_csv('/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba/list_attr_celeba.txt',
                                      delim_whitespace=True, header=None).values[:, 1:],
                          dtype=np.float32)

        labels[labels == -1] = 0

        partition = np.array(pd.read_csv('/srv/node/sdc1/image_data/CelebA/Eval/list_eval_partition.txt', sep=' ',
                                         header=None).values[:, 1], dtype=int)

        file_paths_train = file_paths[partition == 0]
        labels_train = labels[partition == 0]

        nb_exp_train = len(labels_train)

        batch_rand_index = np.random.choice(len(labels_train), size=len(labels_train), replace=False)

        file_paths_train = file_paths_train[batch_rand_index]
        labels_train = labels_train[batch_rand_index]

        file_paths_train = tf.constant(file_paths_train)
        labels_train = tf.constant(labels_train)

        file_paths = file_paths[np.logical_or(partition == 1, partition == 2)]
        labels_val = labels[np.logical_or(partition == 1, partition == 2)]

        np_exp_val = len(labels_val)

        file_paths_val = tf.constant(file_paths)
        labels_val = tf.constant(labels_val)

        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=100000)
        dataset_train = dataset_train.map(map_func=self.parse_image, num_parallel_calls=self.cpu_cores)
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

if __name__ == '__main__':
    vgg = VGGFace()
    train_init, test_init, x, y = vgg.load_dataset(imgs_path='/srv/node/sdc1/image_data/CelebA/Img/img_align_celeba')
    vgg.build(x, y)
    vgg.train(train_init, test_init, n_epochs=1)


