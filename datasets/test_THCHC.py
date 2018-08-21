#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
检查THCHS数据集中是否有词典中没有的词
'''
import re
import json
import time

with open('./char_2_phone_dict.dict', 'r') as f:
    char_2_phone_dict = json.load(f)

with open('./name_THCHS_list.txt', 'r') as f:
    for line in f:
        line = eval(line)[4]
        line = line.strip()
        line = line.split(' ')

        for item in line:
            if item not in char_2_phone_dict.keys():
                print(item)
                time.sleep(0.1)
            if item == 'l':
                print(item)











