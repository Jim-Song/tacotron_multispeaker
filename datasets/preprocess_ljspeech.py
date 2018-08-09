import wave
import os
import sys
sys.path.append('../')
from hparams import hparams as hp



ljspeech_path = '/home/pattern/songjinming/tts/data/LJSpeech-1.0'

with open('name_LJSpeech_list.txt', 'w') as fLJspeech:

    with open(os.path.join(ljspeech_path, 'metadata.csv'), encoding='utf-8') as fCSV:
        ct = 1
        time = 0
        for line in fCSV:
            parts = line.strip().split('|')
            # get sample rate
            wav_path = os.path.join(ljspeech_path, 'wavs', '%s.wav' % parts[0])
            #wav_file = wave.open(wav_path, 'rb')
            #params = wav_file.getparams()
            #sample_rate = params[2]
            #get n frames
            #n_samples = params[3]
            #n_frames = int(n_samples/sample_rate/(hp.frame_shift_ms/1000))

            text = parts[2]
            language = 'English'
            info_list = [wav_path, text, language]
            fLJspeech.write(str(info_list))
            fLJspeech.write('\n')
            print(ct)
            ct += 1
            #time += n_samples / sample_rate
            #print(time)

















