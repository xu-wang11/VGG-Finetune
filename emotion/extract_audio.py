#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/11/4
import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import math
import random
from sklearn.utils import shuffle


def extract_audio(video_root, save_audio_root, save_jpg_root):
    for fname in os.listdir(video_root):
        full_path = os.path.join(video_root, fname)
        save_path = os.path.join(save_audio_root, fname.split('.')[0] + '.wav')

        if not os.path.exists(save_audio_root):
            os.mkdir(save_audio_root)
        if not os.path.exists(save_jpg_root):
            os.mkdir(save_jpg_root)
        cmd = "ffmpeg -i {0} -acodec pcm_s16le -ac 2 {1}".format(full_path, save_path)
        os.system(cmd)
        cmd2 = "ffmpeg -i {0} -vf fps=100/51 {1}_\%04d.jpg".format(full_path, os.path.join(save_jpg_root, fname.split('.')[0]))
        os.system(cmd2)


def audio_feature(audio_path):
    sound_clip, s = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(sound_clip, sr=s, n_mels=64, n_fft=1200, hop_length=int(s * 0.015))
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
    y = tf.placeholder(tf.float32, shape=[None, 7], name='Y')
    # conv1 = conv_layer(x, 'conv1', (5, 5, 1, 16), (16,))
    # pool1 = avg_pool(conv1, 'pool1')
    # conv2 = conv_layer(pool1, 'conv2', (5, 5, 16, 32), (32,))
    # pool2 = avg_pool(conv2, 'pool2')
    #conv3 = conv_layer(pool2, 'conv3', (5, 5, 16, 16), (16,))
    # pool3 = avg_pool(conv3, 'pool3')
    fc1 = fc_layer(x, 'fc1', 512)
    fc1_dropout = tf.nn.dropout(tf.nn.relu(fc1), 0.5)
    fc2 = fc_layer(fc1_dropout, 'fc2', 7)
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


def video_feature(video_path):
    pass

if __name__ == '__main__':
    train_root = "/Users/wangxu/Desktop/Train_AFEW/"
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    emotion_labels = np.identity(7)
    # for emotion in emotions:
    #     extract_audio(os.path.join(train_root, emotion), os.path.join(train_root, emotion + '_audio'), os.path.join(train_root, emotion + '_jpg'))
    train_data = []
    train_label = []
    for i in range(len(emotions)):
        emotion = emotions[i]
        audio_dir = os.path.join(train_root, emotion + '_audio')
        for fname in os.listdir(audio_dir):
            features = np.array(audio_feature(os.path.join(audio_dir, fname)))
            labels = []
            for j in range(features.shape[0]):
                labels.append(emotion_labels[i])
            for f in features:
                train_data.append(f)
            for l in labels:
                train_label.append(l)
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    # ind = [i for i in range(train_data.shape[0])]
    data_set = shuffle(train_data, train_label)
    train_data = data_set[0]
    train_label = data_set[1]
    val_root = "/Users/wangxu/Desktop/Val_AFEW/"
    # emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    # emotion_labels = np.identity(7)
    # for emotion in emotions:
    #     extract_audio(os.path.join(train_root, emotion), os.path.join(train_root, emotion + '_audio'), os.path.join(train_root, emotion + '_jpg'))
    val_data = []
    val_label = []
    for i in range(len(emotions)):
        emotion = emotions[i]
        audio_dir = os.path.join(val_root, emotion + '_audio')
        for fname in os.listdir(audio_dir):
            features = np.array(audio_feature(os.path.join(audio_dir, fname)))
            labels = []
            for j in range(features.shape[0]):
                labels.append(emotion_labels[i])
            for f in features:
                val_data.append(f)
            for l in labels:
                val_label.append(l)
    # min_val = np.min(train_data)
    # max_val = np.max(train_data)
    # train_data = (train_data - min_val) / (max_val - min_val)
    val_data = (val_data - min_val) / (max_val - min_val)

    # build_model(train_data, train_label, val_data, val_label)
    np.save('AFEW_train_input.npz', train_data)
    np.save('AFEW_train_output.npz', train_label)
    np.save('AFEW_test_input.npz', val_data)
    np.save('AFEW_test_output.npz', val_label)




