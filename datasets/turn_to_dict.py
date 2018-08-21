#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
将给定的文本转化为dict
'''
import json

#需要转化的文本文件
txt_file = './lexicon.txt'
#转化后的json文件名
char_2_phone_dict_file = './char_2_phone_THCHS_dict.dict'

char_2_phone_dict = {}

with open(txt_file, 'r') as f:
    for line in f:
        crrt = line.strip().split(' ')
        char_2_phone_dict[crrt[0]] = crrt[1:]

with open(char_2_phone_dict_file, 'w') as f:
    json.dump(char_2_phone_dict, f)



