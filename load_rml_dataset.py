#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/11/11
import os
from pathlib import Path
from shutil import copyfile
import librosa
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import math


def mv_video(root_path):
    if not os.path.exists(root_path / 'video'):
        os.mkdir(root_path / 'video')
    if not os.path.exists(root_path / 'audio'):
        os.mkdir(root_path / 'audio')

    for root, subdirs, files in os.walk(root_path):
        for file in files:
            if file.endswith("avi"):
                relative_dir = Path(root).relative_to(root_path)
                paths = list(relative_dir.parts)
                paths.append(file)
                save_name = '_'.join(paths)
                copyfile(Path(root) / file, root_path / 'video' / save_name)
                cmd = "ffmpeg -i {0} -acodec pcm_s16le -ac 2 {1}".format(Path(root) / file, root_path / 'audio' /
                                                                         (save_name.split('.')[0] + '.wav'))
                os.system(cmd)
                # print(Path(root) / file)


def audio_feature(audio_path):
    sound_clip, s = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(sound_clip, sr=s, n_mels=64, n_fft=1200,
                                              hop_length=int(s * 0.015))
    log_mel_spec = librosa.power_to_db(mel_spec)
    # first_order_log_mel_spec = (log_mel_spec[:, 2:] - log_mel_spec[:, 0:-2]) / 0.03
    # second_order_log_mel_spec = (log_mel_spec[:, 2:] + log_mel_spec[:, 0:-2] - 2 * log_mel_spec[:, 1:-1]) / (0.015 * 0.015)
    static_log_mel_spec = log_mel_spec[:, 1:-1]
    features = []
    for i in range(0, static_log_mel_spec.shape[1] - 63, 34):
        static_feature = static_log_mel_spec[:, i: (i + 64)]
        # first_feature = first_order_log_mel_spec[:, i: (i + 64)]
        # second_feature = second_order_log_mel_spec[:, i: (i + 64)]
        # feature = [static_feature, first_feature, second_feature]
        # feature = np.transpose(feature, [1, 2, 0])
        features.append(np.reshape(static_feature, [64, 64, 1]))
    return features


def load_rml_data(audio_dir):
    emotion = {'an': [1, 0, 0, 0, 0, 0], 'di': [0, 1, 0, 0, 0, 0], 'fe': [0, 0, 1, 0, 0, 0], 'ha': [0, 0, 0, 1, 0, 0],
               'sa': [0, 0, 0, 0, 1, 0], 'su': [0, 0, 0, 0, 0, 1]}
    input = []
    output = []
    for fname in os.listdir(audio_dir):
        features = np.array(audio_feature(os.path.join(audio_dir, fname)))
        labels = []
        emotion_text = fname.split('.')[0].split('_')[-1][0:2]
        for j in range(features.shape[0]):
            labels.append(emotion[emotion_text])
        for f in features:
            input.append(f)
        for l in labels:
            output.append(l)
    return input, output


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name, conv_shape, bias_shape):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        conv = tf.nn.conv2d(bottom, tf.Variable(tf.truncated_normal(conv_shape, stddev=0.1)), [1, 1, 1, 1], padding='SAME')

        bias = tf.nn.bias_add(conv, tf.Variable(tf.constant(0.1, shape=bias_shape)))

        relu = tf.nn.relu(bias)

        return relu


def fc_layer(input, name, output):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = input.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(input, [-1, dim])
        initial = tf.truncated_normal_initializer(0, 1/math.sqrt(dim))
        weights = tf.get_variable('weights', (dim, output), tf.float32, initializer=initial)
        biases = tf.get_variable('biases', (output,), tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def build_model(train_data, label_data, val_data, val_label):
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='X')
    y = tf.placeholder(tf.float32, shape=[None, 6], name='Y')
    conv1 = conv_layer(x, 'conv1', (5, 5, 1, 16), (16,))
    pool1 = avg_pool(conv1, 'pool1')
    conv2 = conv_layer(pool1, 'conv2', (5, 5, 16, 32), (32,))
    pool2 = avg_pool(conv2, 'pool2')
    # conv3 = conv_layer(pool2, 'conv3', (5, 5, 16, 16), (16,))
    # pool3 = avg_pool(conv3, 'pool3')
    fc1 = fc_layer(pool2, 'fc1', 64)
    fc1_dropout = tf.nn.dropout(tf.nn.relu(fc1), 0.5)
    fc2 = fc_layer(fc1_dropout, 'fc2', 6)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc2)
    op_loss = tf.reduce_mean(entropy, name='loss')
    op_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(op_loss)

    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    n_epochs = 300
    batch_size = 300
    num_examples = train_data.shape[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            true_num = 0
            sample_num = 0
            for iteration in range(num_examples // batch_size):
                # this cycle is for dividing step by step the heavy work of each neuron
                X_batch = train_data[iteration * batch_size:iteration * batch_size + batch_size, :, :]
                y_batch = label_data[iteration * batch_size:iteration * batch_size + batch_size]
                sess.run(op_opt, feed_dict={x: X_batch, y: y_batch})
            n = sess.run(correct_prediction, feed_dict={x: train_data, y: label_data})
            true_num = np.sum(n)
            sample_num = train_data.shape[0]
            print("training data: " + str(true_num * 1.0 / sample_num))

            n = sess.run(correct_prediction, feed_dict={x: val_data, y: val_label})
            true_num = np.sum(n)
            sample_num = val_data.shape[0]
            print("validate data: " + str(true_num * 1.0 / sample_num))


if __name__ == '__main__':
    # mv_video(Path('/Users/wangxu/Desktop/RML'))
    # input, output = load_rml_data(Path('/Users/wangxu/Desktop/RML') / 'audio')
    # np.save('rml_audio_feature.npy', input)
    # np.save('rml_label.npy', output)
    input = np.load('rml_audio_feature.npy')
    output = np.load('rml_label.npy')
    min_val = np.min(input)
    max_val = np.max(input)
    input = (input - min_val) / (max_val - min_val)
    # ind = [i for i in range(input.shape[0])]
    data_set = shuffle(input, output)
    train_len = int(data_set[0].shape[0] * 0.7)
    train_data = data_set[0][0:train_len]
    train_label = data_set[1][0:train_len]
    test_data = data_set[0][train_len:]
    test_label = data_set[1][train_len:]
    build_model(train_data, train_label, test_data, test_label)





