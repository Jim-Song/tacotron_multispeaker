import wave
import os
import sys
import re
sys.path.append('../')
from hparams import hparams as hp


dir = '/home/pattern/songjinming/tts/data/arctic'
arctic_dirs = [os.path.join(dir, 'cmu_us_awb_arctic-0.90-release/cmu_us_awb_arctic'),
               os.path.join(dir, 'cmu_us_bdl_arctic-0.95-release/cmu_us_bdl_arctic'),
               os.path.join(dir, 'cmu_us_clb_arctic-0.95-release/cmu_us_clb_arctic'),
               os.path.join(dir, 'cmu_us_jmk_arctic-0.95-release/cmu_us_jmk_arctic'),
               os.path.join(dir, 'cmu_us_ksp_arctic-0.95-release/cmu_us_ksp_arctic'),
               os.path.join(dir, 'cmu_us_rms_arctic-0.95-release/cmu_us_rms_arctic'),
               os.path.join(dir, 'cmu_us_slt_arctic-0.95-release/cmu_us_slt_arctic')
               ]

ct = 1
time = 0
with open('name_arctic_list.txt', 'w') as farctic:
    for dir in arctic_dirs:
        # dir = '/home/pattern/songjinming/tts/data/arctic/cmu_us_slt_arctic-0.95-release/cmu_us_slt_arctic'
        with open(os.path.join(dir, 'etc', 'txt.done.data')) as ftext:
            #'/home/pattern/songjinming/tts/data/arctic/cmu_us_slt_arctic-0.95-release/cmu_us_slt_arctic/etc/txt.done.data'
            wav_dir = os.path.join(dir, 'wav')
            # wav_dir = '/home/pattern/songjinming/tts/data/arctic/cmu_us_slt_arctic-0.95-release/cmu_us_slt_arctic/wav'
            for line in ftext:
                #line = ( arctic_a0001 "AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC" )
                print(line)
                name_file = re.findall('arctic_[ab][0-9]+', line)[0]#'arctic_a0001'
                text = re.findall('"(.*)"', line)[0]#'AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC'
                # get sample rate
                wav_path = os.path.join(dir, 'wav', '%s.wav' % name_file)
                #wav_path = '/home/pattern/songjinming/tts/data/arctic/cmu_us_awb_arctic-0.90-release/
                #           cmu_us_awb_arctic/wav/arctic_a0001.wav'
                wav_file = wave.open(wav_path, 'rb')
                params = wav_file.getparams()
                sample_rate = params[2]
                # get n frames
                n_samples = params[3]
                language = 'English'
                info_list = [wav_path, text, language, sample_rate, n_samples]
                farctic.write(str(info_list))
                farctic.write('\n')
                print(ct)
                ct += 1
                time += n_samples/sample_rate
                print(time)















