import wave
import os
import fnmatch
import sys
import re
import os

sys.path.append('../')
from hparams import hparams as hp

FILE_PATTERN = r'([0-9]+)_([0-9]+)_([0-9]+)\.wav'



VCTK_path = '/home/pattern/songjinming/tts/data/VCTK-Corpus'
VCTK_path_wav = os.path.join(VCTK_path, 'wav48')
VCTK_path_txt = os.path.join(VCTK_path, 'txt')

ct = 1
time = 0
with open('name_VCTK_list.txt', 'w') as fVCTK:
    for dir_name in os.listdir(VCTK_path_wav):
        #dir_name = 'p300'
        #wav_dir  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/wav48/p300'
        #txt_dir  = ''/home/pattern/songjinming/tts/data/VCTK-Corpus/txt/p300''
        wav_dir = os.path.join(VCTK_path_wav, dir_name)
        txt_dir = os.path.join(VCTK_path_txt, dir_name)
        for wav_file in os.listdir(wav_dir):
            #wav_file  = 'p300_224.wav'
            #name_file = 'p300_224'
            #txt_file  = 'p300_224.txt'
            #wav_root  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/wav48/p300/p300_224.wav'
            #txt_root  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/txt/p300/p300_224.txt'
            name_file = os.path.splitext(wav_file)[0]
            #some file is not wav file and just skip
            #print(os.path.splitext(wav_file))
            if not os.path.splitext(wav_file)[1] == '.wav':
                continue
            txt_file = '.'.join([name_file, 'txt'])
            wav_root = os.path.join(wav_dir, wav_file)
            txt_root = os.path.join(txt_dir, txt_file)
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
                text = re.sub('\n', '', text)
            except:
                continue
            #write
            language = 'English'
            info_list = [wav_root, text, language]
            fVCTK.write(str(info_list))
            fVCTK.write('\n')
            ct += 1
            #time += n_samples / sample_rate
            #print(time)

            print(ct)
























