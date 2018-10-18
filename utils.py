#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/10/18
import tensorflow as tf
import numpy as np
import os
import pandas as pd


def parse_image_net_image(filename, label):
    averageImg_BGR = tf.constant(
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
    image_nml = image_bgr - averageImg_BGR
    label = tf.one_hot(indices=label, depth=1000)
    return image_nml, label


def parse_image_net_image_for_train(file_name, label):
    averageImg_BGR = tf.constant(
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
    image_nml = image_bgr - averageImg_BGR
    image_cropped = tf.image.random_flip_left_right(tf.random_crop(image_nml, [224, 224, 3]))
    label = tf.one_hot(indices=label, depth=1000)
    return image_cropped, label


def load_image_net_dataset(imgs_path, label_path, cpu_cores, batch_size):
    # % load data
    img_dir = imgs_path
    file_paths = np.array([os.path.join(img_dir,  x) for x in sorted(os.listdir(img_dir))])

    labels = np.array(pd.read_csv(label_path, delim_whitespace=True, header=None).values[:, 1], dtype=np.int32)

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


    file_paths_train = tf.constant(file_paths_train)
    labels_train = tf.constant(labels_train)

    file_paths_val = tf.constant(file_paths_val)
    labels_val = tf.constant(labels_val)

    # %% construct input pipeline

    dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
    dataset_train = dataset_train.shuffle(buffer_size=100000)
    dataset_train = dataset_train.repeat(count=4)
    dataset_train = dataset_train.map(map_func=parse_image_net_image_for_train, num_parallel_calls=cpu_cores)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(buffer_size=1)

    dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
    dataset_val = dataset_val.map(map_func=parse_image_net_image, num_parallel_calls=cpu_cores)
    dataset_val = dataset_val.batch(batch_size)
    dataset_val = dataset_val.prefetch(buffer_size=1)

    vgg_iter = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    x, y = vgg_iter.get_next()

    train_init = vgg_iter.make_initializer(dataset_train)  # initializer for train_data
    test_init = vgg_iter.make_initializer(dataset_val)
    # return dataset_train, dataset_val, nb_exp_train, np_exp_val
    return train_init, test_init, x, y


def parse_celeba_image(filename, label):
    average_face = tf.constant(
        np.expand_dims(np.expand_dims(np.array([93.5940, 104.7624, 129.1863], dtype=np.float32), axis=0), axis=0))
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)
    image_bgr = tf.reverse(image_resized, axis=[-1])
    image_nml = image_bgr - average_face
    return image_nml, label


def load_face_dataset(imgs_dir, attr_file, partition_file, cpu_cores, batch_size):
    file_paths = np.array([os.path.join(imgs_dir, x) for x in sorted(os.listdir(imgs_dir))])

    labels = np.array(pd.read_csv(attr_file, delim_whitespace=True, header=None).values[:, 1:], dtype=np.float32)

    labels[labels == -1] = 0

    partition = np.array(pd.read_csv(partition_file, sep=' ', header=None).values[:, 1], dtype=int)

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
    dataset_train = dataset_train.map(map_func=parse_celeba_image, num_parallel_calls=cpu_cores)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(buffer_size=1)

    dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
    dataset_val = dataset_val.map(map_func=parse_celeba_image, num_parallel_calls=cpu_cores)
    dataset_val = dataset_val.batch(batch_size)
    dataset_val = dataset_val.prefetch(buffer_size=1)

    vgg_iter = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    x, y = vgg_iter.get_next()

    train_init = vgg_iter.make_initializer(dataset_train)  # initializer for train_data
    test_init = vgg_iter.make_initializer(dataset_val)
    # return dataset_train, dataset_val, nb_exp_train, np_exp_val
    return train_init, test_init, x, y