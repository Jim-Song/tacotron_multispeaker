#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
检查THCHS数据集中是否有词典中没有的词
'''
import re
import json
import time

with open('./char_2_phone_dict.json', 'r') as f:
    char_2_phone_dict = json.load(f)

with open('./normal.json', 'r') as f:
    normal = json.load(f)

with open('./name_THCHS_list.txt', 'r') as f:
    for line in f:
        line = eval(line)[1]
        line = line.strip()
        line = line.split(' ')

        for item in line:
            if item not in char_2_phone_dict.keys():
                print(item)

with open('./name_THCHS_list.txt', 'r') as f:
    for line in f:
        line = eval(line)[1]
        line = line.strip()
        line = line.split(' ')

        for item in line:
            for item2 in item:
                if item2 not in normal:
                    print(item2)












