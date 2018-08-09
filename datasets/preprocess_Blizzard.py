import wave
import os
import sys
sys.path.append('../')
from hparams import hparams as hp



blizzard_path = '/home/pattern/songjinming/tts/data/Blizzard'
_min_confidence = 90

books = [
  'ATrampAbroad',
  'TheManThatCorruptedHadleyburg',
  'LifeOnTheMississippi',
  'TheAdventuresOfTomSawyer',
]

ct = 1
time = 0
with open('name_Blizzard_list.txt', 'w') as fBlizzard:
    for book in books:
        with open(os.path.join(blizzard_path, book, book, 'sentence_index.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if line[0] is not '#' and len(parts) == 8 and float(parts[3]) > _min_confidence:


                    text = parts[5]

                    # get sample rate
                    wav_path = os.path.join(blizzard_path, book, 'wav', '%s.wav' % parts[0])
                    #wav_file = wave.open(wav_path, 'rb')
                    #params = wav_file.getparams()
                    #sample_rate = params[2]
                    # get n frames
                    #n_samples = params[3]
                    #n_frames = int(n_samples / sample_rate / (hp.frame_shift_ms / 1000))
                    language = 'English'
                    info_list = [wav_path, text, language]
                    fBlizzard.write(str(info_list))
                    fBlizzard.write('\n')
                    print(ct)
                    ct += 1
                    #time += n_samples / sample_rate
                    #print(time)

            