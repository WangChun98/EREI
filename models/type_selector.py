# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/24 17:04
# @Author : WangChun

import numpy as np
import keras
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input, Lambda
from keras import metrics
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Model
from keras.models import load_model


# 指定程序在某个GPU上运行
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def count_lines(filename):
    with open(filename, encoding="utf-8") as f:
        count = 0
        for line in f.readlines():
            count += 1
    print(count)
    return count

def read_dataset_x(filename):
    all_texts = []
    file_len = count_lines(filename)
    with open(filename, encoding="utf-8") as f:
        i = 0
        while i < file_len:
        # while i <= 10:
            line = f.readline()
            line_list = line.strip().split()
            temp_list = line_list[1:]
            temp_line = ' '.join(temp_list)
            all_texts.append(temp_line)
            i += 1
    return all_texts
def read_dataset_x2(filename):
    all_texts = []
    file_len = count_lines(filename)
    with open(filename, encoding="utf-8") as f:
        i = 0
        while i < file_len:
        # while i <= 10:
            line = f.readline().strip()
            all_texts.append(line)
            i += 1
    return all_texts
def read_dataset_y(filename):
    all_labels = []
    file_len = count_lines(filename)
    with open(filename, encoding="utf-8") as f:
        i = 0
        while i < file_len:
        # while i <= 10:
            line_new = []
            line = f.readline().strip()

            for item in line:
                item = int(item)
                line_new.append(item)

            all_labels.append(line_new)
            i += 1
    return all_labels
def read_dataset_y2(filename):
    all_labels = []
    file_len = count_lines(filename)
    with open(filename, encoding="utf-8") as f:
        i = 0
        while i < file_len:
            line = f.readline().strip()
            all_labels.append(line)
            i += 1
    return all_labels

train_x_filename = 'mojitalk_data/train.ori'
train_y_filename = 'processing/label_data/train_label.txt'

dev_x_filename = 'mojitalk_data/dev.ori'
dev_y_filename = 'processing/label_data/train_label.txt'

test_x_filename = 'mojitalk_data/train.rep'
test_y_filename = 'processing/label_data/train_label.txt'

x_tr = read_dataset_x(train_x_filename)
y_tr = read_dataset_y2(train_y_filename)
x_val = read_dataset_x(dev_x_filename)
y_val = read_dataset_y2(dev_y_filename)
x_test = read_dataset_x2(test_x_filename)
y_test = read_dataset_y2(test_y_filename)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_tr))
x_tr_seq = tokenizer.texts_to_sequences(x_tr)
x_val_seq = tokenizer.texts_to_sequences(x_val)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_tr_seq = pad_sequences(x_tr_seq, maxlen=50, value=0, padding='post')
x_val_seq = pad_sequences(x_val_seq, maxlen=50, value=0, padding='post')
x_test_seq = pad_sequences(x_test_seq, maxlen=50, value=0, padding='post')
print('x_tr_seq_padding:', x_tr_seq)

size_of_vocabulary = len(tokenizer.word_index) + 1
print('size_of_vocabulary:', size_of_vocabulary)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./image/rep_label_auto.png'.format(train))

def LSTM_new(x_train_padded_seqs, y_train, x_dev_padded_seqs, y_dev, x_test_padded_seqs, y_test, file_label_predict):
    model = Sequential()
    model.add(Embedding(size_of_vocabulary, 128, input_length=50, trainable=False))
    # LSTM
    model.add(Dropout(0.2))
    model.add(LSTM(units=16))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('./save/ori_label_balanced.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)
    val_hot_labels = keras.utils.to_categorical(y_dev, num_classes=2)
    class_weight = 'balanced'
    history = model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=50, class_weight=class_weight,
                        validation_data=(x_dev_padded_seqs, val_hot_labels), callbacks=[es, mc])

    model.save('./save/ori_label_balanced.h5')
    model = load_model('./save/ori_label_test2.h5')

    y_test_onehot = keras.utils.to_categorical(y_test, num_classes=2)  #
    y_test_onehot_results = np.argmax(y_test_onehot, axis=1)

    result = model.predict(x_test_padded_seqs)
    result_labels = np.argmax(result, axis=1)
    y_predict = list(map(str, result_labels))
    return y_predict

file_pre_label = "./train_rep_label_balanced.txt"
y_pre = LSTM_new(np.array(x_tr_seq), y_tr,np.array(x_val_seq), y_val, np.array(x_test_seq), y_test, file_pre_label)
with open(file_pre_label, 'w', encoding='utf-8') as f:
    for i in range(0, len(y_pre)):
        f.write(y_pre[i])
        f.write("\n")
f.close()
