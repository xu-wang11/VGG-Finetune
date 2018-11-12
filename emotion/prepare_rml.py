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
    for root, subdirs, files in os.walk(Path(root_path) / 'raw'):

        for file in files:
            if file.endswith("avi"):
                relative_dir = Path(root).relative_to(root_path / 'raw')
                paths = list(relative_dir.parts)
                paths.append(file)
                save_name = '_'.join(paths)
                copyfile(Path(root) / file, root_path / 'video/' / save_name)
                cmd = "ffmpeg -i {0} -acodec pcm_s16le -ac 2 {1}".format(Path(root) / file, root_path / 'audio' /
                                                                         (save_name.split('.')[0] + '.wav'))
                os.system(cmd)
                # print(Path(root) / file)


def audio_feature(audio_path):
    print(audio_path)
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
    train_user = ['s1', 's2', 's3', 's4', 's5', 's6']
    val_user = ['s7', 's8']
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for fname in os.listdir(audio_dir):
        if str(fname).endswith('wav'):
            features = np.array(audio_feature(os.path.join(audio_dir, fname)))
            labels = []
            emotion_text = fname.split('.')[0].split('_')[-1][0:2]
            train_or_val = fname.split('.')[0].split('_')[0] in train_user
            for j in range(features.shape[0]):
                labels.append(emotion[emotion_text])
            for f in features:
                if train_or_val:
                    train_input.append(f)
                else:
                    test_input.append(f)
            for l in labels:
                if train_or_val:
                    train_output.append(l)
                else:
                    test_output.append(l)

    return train_input, train_output, test_input, test_output


if __name__ == '__main__':
    # mv_video(Path('/Users/wangxu/Desktop/RML'))
    train_input, train_output, test_input, test_output = load_rml_data(Path('/Users/wangxu/Desktop/RML') / 'audio')
    np.save('rml_train_input.npy', train_input)
    np.save('rml_train_output.npy', train_output)

    np.save('rml_test_input.npy', test_input)
    np.save('rml_test_output.npy', test_output)






