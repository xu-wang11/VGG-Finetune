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





if __name__ == '__main__':
    # mv_video(Path('/Users/wangxu/Desktop/RML'))
    input, output = load_rml_data(Path('/Users/wangxu/Desktop/RML') / 'audio')
    # np.save('rml_audio_feature.npy', input)
    # np.save('rml_label.npy', output)






