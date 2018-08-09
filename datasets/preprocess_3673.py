import wave
import os
import fnmatch
import sys
import re
import os

sys.path.append('../')
from hparams import hparams as hp




f3673_path = '/home/pattern/songjinming/tts/data/all_novel_data/3673'
f3673_wav = os.path.join(f3673_path, 'wav')
f3673_txt = os.path.join(f3673_path, 'stm2')

ct = 1
time = 0
with open('name_f3673_list.txt', 'w') as f3673:

    for wav_file in os.listdir(f3673_wav):
        #wav_file  = '03673_01768_00132.wav'
        #name_file = '03673_01768_00132'
        #txt_file  = '03673_01768_00132.stm'
        #wav_root  = '/home/pattern/songjinming/tts/data/all_novel_data/3673/wav/03673_01768_00132.wav'
        #txt_root  = '/home/pattern/songjinming/tts/data/all_novel_data/3673/stm2/03673_01768_00132.stm'
        name_file = os.path.splitext(wav_file)[0]
        #some file is not wav file and just skip
        #print(os.path.splitext(wav_file))
        if not os.path.splitext(wav_file)[1] == '.wav':
            continue
        txt_file = '.'.join([name_file, 'stm'])
        wav_root = os.path.join(f3673_wav, wav_file)
        txt_root = os.path.join(f3673_txt, txt_file)
        #audio
        #print(wav_root)
        #f_wav = wave.open(wav_root, 'rb')
        #params = f_wav.getparams()
        #sample_rate = params[2]
        #n_samples = params[3]
        #txt
        #some wav files dont have correspond txt file
        try:
            text = open(txt_root, 'r').read()
            #text = re.sub('\n', '', text)
        except:
            continue
        #write
        language = 'Chinese'
        info_list = [wav_root, text, language]
        f3673.write(str(info_list))
        f3673.write('\n')
        ct += 1
        #time += n_samples / sample_rate
        #print(time)

        print(ct)
