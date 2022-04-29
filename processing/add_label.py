# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/24 17:04
# @Author : WangChun
import sys
import io
import os
import linecache
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def get_emoji_codes(filename):
    emoji_dict = {}
    emoji_64_dict = {}

    with open(filename, encoding="utf-8") as f:
        line = f.readline()
        while line != "":
            if line.strip() == "" or line.startswith('#'):
                line = f.readline()
                continue
            if not(line.startswith('1') or line.startswith('2')):
                line = f.readline()
                continue
            emoji, desc = line.strip().split('# ')[-1].split(' ', 1)
            emoji_dict[emoji] = desc
            if desc.endswith(" 64"):
                emoji_64_dict[emoji] = desc
            line = f.readline()

    return emoji_dict, emoji_64_dict

def count_lines(filename):
    with open(filename, encoding="utf-8") as f:
        count = 0
        for line in f.readlines():
            count += 1
    print(count)
    return count

def get_line(file, nums_line):
    return linecache.getline(file, nums_line).strip()

def add_exist_emojis_label(filename, filename_new, emojis):
    filename_len = count_lines(filename)
    f_new = open(filename_new, "w", encoding='utf-8')
    with open(filename, encoding="utf-8") as f:
        # while line != "":
        i = 0
        j = 0
        m = 0
        count_1 = 0
        while i < filename_len:
        # while i <= 40:
            line = f.readline()
            line_without_label = line[1:].strip()
            line_without_label_list = line_without_label.split()
            flag = 0
            for em in emojis:
                for j in range(len(line_without_label_list)):
                    if line_without_label_list[j] == em:
                        flag = 1
            if flag == 1:
                label = 1
                count_1 = count_1 + 1
            else:
                label = 0
            f_new.write(str(label))
            f_new.write('\n')
            i = i + 1
    f_new.close()
    print(count_1)

emoji_dict, emoji_64_dict = get_emoji_codes("emoji-test.txt")
emojis = list(emoji_dict.keys())
filename_train = 'mojitalk_data/train.rep'
filename_dev = 'mojitalk_data/dev.rep'
filename_test = 'mojitalk_data/test.rep'
train_label = './label_data/train_label.txt'
add_exist_emojis_label(filename_train, train_label, emojis)



