#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/11/
from vgg_base import VGGBase
import numpy as np
import os
import librosa
from random import shuffle
import tensorflow as tf
import time
from scipy.misc import imresize


class AudioEmotion(VGGBase):

    @staticmethod
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

    def audio_dataset(self):
        train_root = "/srv/node/sdc1/Train_AFEW"
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
                features = np.array(self.audio_feature(os.path.join(audio_dir, fname)))
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
        train_label = np.array(train_label)
        # shuffle(train_data, train_label)

        val_root = "/srv/node/sdc1/Val_AFEW"
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
                features = np.array(self.audio_feature(os.path.join(audio_dir, fname)))
                labels = []
                for j in range(features.shape[0]):
                    labels.append(emotion_labels[i])
                for f in features:
                    val_data.append(f)
                for l in labels:
                    val_label.append(l)

        val_data = (val_data - min_val) / (max_val - min_val)
        val_label = np.array(val_label)
        return train_data, train_label, val_data, val_label

    def __init__(self):
        super().__init__()

        self.batch_size = 128
        self.cpu_cores = 8
        self.skip_step = 100
        self.lr = 0.01

        self.op_opt = None
        self.op_loss = None
        self.op_summary = None
        self.accuracy = None
        self.preds = None
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 1], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, 7], name='Y')

    # override fc layer for CelebA dataset
    def fc_layers(self, input):
        fc6 = self.fc_layer(input, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        logits = self.fc_layer(relu7, "fc8")

        return logits

    def loss(self, labels, logits):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.op_loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        var_list = self.trainable_variables()
        self.op_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.op_loss, var_list=self.trainable_variables(), global_step=self.global_step)

    def trainable_variables(self):
        # var_list = [v for v in tf.trainable_variables() if v.name.startswith("fc")]
        var_list = [v for v in tf.trainable_variables()]
        return var_list

    def prediction(self, labels, logits):
        with tf.name_scope('predict'):
            print(labels.shape)
            print(logits.shape)
            predictions = tf.nn.softmax(logits)
            self.preds = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.op_loss)
            # tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.op_loss)
            self.op_summary = tf.summary.merge_all()

    def build(self, x, y):
        logits = self.inference(x)
        self.loss(y, logits)
        self.optimize()
        self.prediction(y, logits)
        self.summary()

    def train(self, n_epochs, train_data, label_data, test_data, test_label):
        writer = tf.summary.FileWriter('graphs/vgg_net', tf.get_default_graph())
        num_examples = train_data.shape[0]
        batch_size = self.batch_size
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            self.save_model(sess, 'vgg_net_before_train.data')
            step = self.global_step.eval()
            for epoch in range(n_epochs):
                true_num = 0
                sample_num = 0
                for iteration in range(num_examples // batch_size):
                    # this cycle is for dividing step by step the heavy work of each neuron
                    X_batch = train_data[iteration * batch_size:iteration * batch_size + batch_size, :, :]
                    y_batch = label_data[iteration * batch_size:iteration * batch_size + batch_size]
                    sess.run(self.op_opt, feed_dict={self.x: X_batch, self.y: y_batch})
                for iteration in range(test_data.shape[0] // batch_size):
                    X_batch = test_data[iteration * batch_size:iteration * batch_size + batch_size, :, :]
                    y_batch = test_label[iteration * batch_size:iteration * batch_size + batch_size]

                    n = sess.run(self.preds, feed_dict={self.x: X_batch, self.y: y_batch})
                    true_num += np.sum(n)
                    sample_num += X_batch.shape[0]
                print("Validating data: " + str(true_num * 1.0 / sample_num))


            self.save_model(sess, 'vgg_net_final_train.data')
        writer.close()


if __name__ == '__main__':
    vgg = AudioEmotion()

    train_data, train_label, val_data, val_label = vgg.audio_dataset()

    resize_train_data = []
    for i in range(train_data.shape[0]):
        img = train_data[i, :, :, 0]
        # img = np.reshape(img, (64, 64))
        new_img = imresize(img, (224, 224))
        new_img = np.reshape(new_img, (224, 224, 1))
        resize_train_data.append(new_img)
    resize_train_data = np.array(resize_train_data)

    resize_val_data = []
    for i in range(val_data.shape[0]):
        img = val_data[i, :, :, 0]
        # img = np.reshape(img, (64, 64))
        new_img = imresize(img, (224, 224))
        new_img = np.reshape(new_img, (224, 224, 1))
        resize_val_data.append(new_img)
    resize_val_data = np.array(resize_val_data)
    vgg.load_model(model_path='vgg_result/audio_1.model')
    vgg.build(vgg.x, vgg.y)

    vgg.train(100, resize_train_data, train_label, resize_val_data, val_label)

