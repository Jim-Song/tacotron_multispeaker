import re
import json
import time

with open('./char_2_phone_dict.dict', 'r') as f:
    char_2_phone_dict = json.load(f)

with open('./aishell_transcript_v0.8.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(' ')
        line = line[1:]

        for item in line:
            if item == '':
                continue
            if item not in char_2_phone_dict.keys():
                print(item)
                print(line)


