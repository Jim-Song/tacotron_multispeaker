import re
import json
import time

with open('./char_2_phone_dict.json', 'r') as f:
    char_2_phone_dict = json.load(f)

with open('./normal.json', 'r') as f:
    normal = json.load(f)

with open('../../data/data_aishell/transcript/aishell_transcript_v0.8.txt', 'r') as f:
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


with open('../../data/data_aishell/transcript/aishell_transcript_v0.8.txt', 'r') as f:
    ct = 0
    auxilary = []
    for line in f:
        line = line.strip()
        line = line.split(' ')
        line = line[1:]

        for item in line:
            for item2 in item:
                if item2 == '':
                    continue
                if item2 not in normal:
                    #print(item2)
                    ct += 1
                    if ct % 100 == 0:
                        print(ct)
                    if item2 not in auxilary:
                        auxilary.append(item2)
                        print(item2)
with open('auxilary_aishell.txt', 'w') as f:
    for item in auxilary:
        f.write(item)
        f.write('\n')