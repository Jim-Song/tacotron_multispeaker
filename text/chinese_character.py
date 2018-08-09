import os
import re
import json

path = '/home/pattern/songjinming/tts/data/all_novel_data/3673'
stm_path = '/home/pattern/songjinming/tts/data/all_novel_data/3673/stm'
stm2_path = '/home/pattern/songjinming/tts/data/all_novel_data/3673/stm2'
_curly_re = re.compile(r'(.*?)(<.+?>)(.*)')
character_list = []

os.makedirs(stm2_path, exist_ok=True)

ct = 0

for item in os.listdir(stm_path):
    stm_file = os.path.join(stm_path, item)
    with open(stm_file, 'r') as f:
        content = f.read()
    content = _curly_re.match(content)
    content = content.group(3)
    content = content.split(' ')
    content = ''.join(content)

    for character in content:
        if not character in character_list:
            character_list.append(character)

    with open(os.path.join(stm2_path, item), 'w') as f:
        f.write(content)

    ct += 1
    if ct % 10000 == 0:
        print(ct)

with open(os.path.join(path, 'character_list.json'), 'w') as f:
    json.dump(character_list, f)

print(character_list)









