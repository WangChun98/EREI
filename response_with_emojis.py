#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/24 17:04
# @Author : WangChun
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import io
from models.emotion_detector.emotion_track.global_variables import PRETRAINED_PATH, VOCAB_PATH
import json
import csv
import numpy as np
from models.emotion_detector.emotion_track.sentence_tokenizer import SentenceTokenizer
from models.emotion_detector.emotion_track.model_def import deepmoji_emojis
import emoji
import math
import string

EMOJIS = ":face_with_tears_of_joy: :unamused_face: :weary_face: :loudly_crying_face: :smiling_face_with_heart-eyes: :pensive_face: :OK_hand: :smiling_face_with_smiling_eyes: :red_heart: :smirking_face: :beaming_face_with_smiling_eyes: :musical_notes: :flushed_face: :hundred_points: :sleeping_face: :relieved_face: :smiling_face: :raising_hands: :two_hearts: :expressionless_face: :grinning_face_with_sweat: :folded_hands: :confused_face: :face_blowing_a_kiss: :heart_suit: :neutral_face: :person_tipping_hand: :disappointed_face: :see-no-evil_monkey: :tired_face: :victory_hand: :smiling_face_with_sunglasses: :pouting_face: :thumbs_up: :crying_face: :sleepy_face: :face_savoring_food: :face_with_steam_from_nose: :raised_hand: :face_with_medical_mask: :clapping_hands: :eyes: :water_pistol: :persevering_face: :smiling_face_with_horns: :downcast_face_with_sweat: :broken_heart: :heart_decoration: :headphone: :speak-no-evil_monkey: :winking_face: :skull: :confounded_face: :grinning_face_with_smiling_eyes: :winking_face_with_tongue: :angry_face: :person_gesturing_NO: :flexed_biceps: :oncoming_fist: :purple_heart: :sparkling_heart: :blue_heart: :grimacing_face: :sparkles:".split(" ")
maxlen = 30
EmoTagFile = "./EmoTag_64.csv"
punc = string.punctuation

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
# load model
print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)


def sentense_to_n_emojis(sentense):
    i = 0
    j = 0
    temp_sen_add = []
    temp_sen_add_sentense = []
    for j in range(0, len(sentense.split())):
        temp_sen = sentense.split()[j]
        temp_sen_add.append(temp_sen)
        str = ' '.join(temp_sen_add)
        temp_sen_add_sentense.append(str)
    return temp_sen_add_sentense

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def predict_post_emoji(post):
    TEST_SENTENCES = sentense_to_n_emojis(post)
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model.predict(tokenized)
    scores = []

    emoji_vec = []
    # i：索引，t句子
    for i, t in enumerate(TEST_SENTENCES):
        t_tokens = tokenized[i]
        # [18   87 3497 1864   19    0    0    0    0    0    0    0    0    0
        #  0    0    0    0    0    0    0    0    0    0    0    0    0    0
        #  0    0]
        t_score = []

        t_prob = prob[i]

        ind_top = top_elements(t_prob, 1)

        emojis = map(lambda x: EMOJIS[x], ind_top)
        t_score.extend(ind_top)

        emoji_vec_temp = emoji.emojize("{}".format(' '.join(emojis)), use_aliases=True)
        emoji_vec.append(emoji_vec_temp)
    return emoji_vec

def len_list(result_emoji_list, emoji_label):
    emoji_label_list = []
    for i in range(0, len(result_emoji_list)):
        emoji_label_list.append(emoji_label)
    return emoji_label_list

def find_current_emoji(post_emoji, label_emoji):
    current_emoji_list = []
    for i in range(0, len(post_emoji)):
        temp_current_emoji = find_Emoji(post_emoji[i], label_emoji[i])
        current_emoji_list.append(temp_current_emoji)
    return current_emoji_list

def emotionlist(tempemoji):
    data = pd.read_csv(EmoTagFile, index_col='emoji')
    temp_eight_vec = list(data.loc[tempemoji])[2:]
    return temp_eight_vec

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

def minDist(tempemotion):
    data = pd.read_csv(EmoTagFile)
    h = data.shape[0] # 行数
    # print(h)
    min = 100
    for i in range(h):
        rowlist = list(data.loc[i])[3:]
        tempdist = eucliDist(tempemotion, rowlist)
        if min > tempdist:
            min = tempdist
            emoji = list(data.loc[i])[1]
    return emoji, min
def find_Emoji(emoji1, emoji2):
    vec1 = emotionlist(emoji1)
    vec1 = np.array(vec1) # b
    vec1 = vec1 * 9
    vec2 = emotionlist(emoji2) # a
    vec2 = np.array(vec2)
    vec2 = vec2 * 10

    tempEmotionVec = np.multiply(vec1, vec2) # type  <class 'numpy.ndarray'>
    tempEmotionVec = list(tempEmotionVec)
    emoji_finded, _ = minDist(tempEmotionVec)
    return emoji_finded

def instead_punc(out, current_emoji_list):
    punc_list = []
    for index, item in enumerate(out.split()):
        if item in punc:
            punc_list.append(index)
            continue

    out_list = out.split()
    if len(punc_list) != 0:
        if len(out_list) <= len(current_emoji_list):
            for index, item_o in enumerate(out_list):
                if index in punc_list:
                    out_list[index] = current_emoji_list[index]
        if len(out_list) > len(current_emoji_list):
            for i in range(0, len(current_emoji_list)):
                if i in punc_list:
                    out_list[i] = current_emoji_list[i]
            for i in range(len(current_emoji_list), len(out_list)):
                if i in punc_list:
                    out_list[i] = current_emoji_list[-1]
    else:
        out_list.append(current_emoji_list[-1])
    new_out = ' '.join(out_list)
    return new_out

def write_file(file_ori, file_out, emoji_list_file, file_out_new):
    f1 = open(file_ori, "r", encoding='utf-8')
    f2 = open(file_out, "r", encoding='utf-8')
    # f3 = open(emoji_list_file, "w", encoding='utf-8')
    f4 = open(file_out_new, "w", encoding='utf-8')
    linelist1 = f1.readlines()
    linelist2 = f2.readlines()
    # for line in linelist:
    for i in range(0, len(linelist1)):
        line = linelist1[i].strip()
        line_list = line.split()
        temp_list = line_list[1:]
        temp_line = ' '.join(temp_list)
        label = line_list[0]
        emoji_post = predict_post_emoji(temp_line)
        emoji_label = len_list(emoji_post, label)
        current_list = find_current_emoji(emoji_post, emoji_label)
        new_out = instead_punc(linelist2[i].strip(), current_list)
        f4.write(str(new_out))
        f4.write("\n")
    f1.close()
    f2.close()
    # f3.close()
    f4.close()