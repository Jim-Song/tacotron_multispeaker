import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import sys
sys.path.append('../')
from util import audio
import re
import json
from hparams import hparams
import time
import os, numpy, argparse, random, time
import multiprocessing


def _process_utterance(wav_path, text, id):
    '''Preprocesses a single utterance audio/text pair.
    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.
    Args:
      wav_path: Path to the audio file containing the speech input
      seq: The text in the input audio file
      id : identity
    Returns:
      A example containing many datas
    '''
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32).T
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    return wav, spectrogram, mel_spectrogram, text, id


def write_worker(q_out, npy_dir, data_name, id_num):
    pre_time = time.time()
    count = 1
    with open(os.path.join('./datasets', 'npy_input_' + data_name + '_id_num_' + str(id_num) + '.txt'), 'w') as f:
        while True:
            deq = q_out.get()
            if deq is None:
                break
            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, 'count:', count)
                pre_time = cur_time
            count += 1
            (wav, spectrogram, mel_spectrogram, text, id) = deq

            spectrogram_filename = os.path.join(npy_dir, data_name + '-spec-%05d.npy' % count)
            mel_filename = os.path.join(npy_dir, data_name + '-mel-%05d.npy' % count)
            wav_filename = os.path.join(npy_dir, data_name + '-wav-%05d.npy' % count)
            np.save(spectrogram_filename, spectrogram, allow_pickle=False)
            np.save(mel_filename, mel_spectrogram, allow_pickle=False)
            np.save(wav_filename, wav, allow_pickle=False)

            info_list = [spectrogram_filename, mel_filename, wav_filename, text, id]
            f.write(str(info_list) + '\n')


def audio_encoder(item, q_out):
    wav_root = item[0]
    text = item[1]
    id = item[2]
    wav, spectrogram, mel_spectrogram, text, id = _process_utterance(wav_root, text, id)
    q_out.put([wav, spectrogram, mel_spectrogram, text, id])


def read_worker(q_in, q_out):
    while True:
        item = q_in.get()
        if item is None:
            break
        audio_encoder(item, q_out)


def wav_to_npy_read_from_text(args, text_path, data_name, id_num):
    npy_dir = os.path.join(args.output, "npy_tacotron_" + data_name + '_id_num_' + str(id_num))
    os.makedirs(npy_dir, exist_ok=True)

    q_in = [multiprocessing.Queue(1024) for i in range(args.num_workers)]
    q_out = multiprocessing.Queue(1024)
    read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) for i in range(args.num_workers)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, npy_dir, data_name, id_num))
    write_process.start()
    with open(text_path, 'r') as f:
        ct = 0
        for line in f:
            line = eval(line)
            q_in[ct % len(q_in)].put(line)
            ct += 1
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()
    q_out.put(None)
    write_process.join()

    try:
        with open('./train_npy_data_dict.json', 'r') as f:
            train_data_dict = json.load(f)
    except:
        train_data_dict = {}
    train_data_dict[data_name] = os.path.join('./datasets', 'npy_input_' + data_name + '_id_num_' + str(id_num) + '.txt')
    with open('./train_npy_data_dict.json', 'w') as f:
        json.dump(train_data_dict, f)













