import tensorflow as tf
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



with open('./datasets/phone+character+alphabet+punctuation_list.json', 'r') as f:
    _symbols = json.load(f)



_pad = '_'
_eos = '~'

symbols = [_pad, _eos] + _symbols
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


#generate the int
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#generate the str typt
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _symbols_to_sequence(symbols):
    out = []
    for s in symbols:
        try:
            out.append(_symbol_to_id[s])
        except:
            out.append(_symbol_to_id['_'])
    return out



def _process_utterance(wav_path, seq, id):
    '''Preprocesses a single utterance audio/text pair.
    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.
    Args:
      wav_path: Path to the audio file containing the speech input
      seq: The text spoken in the input audio file
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

    input_lengths = len(seq)
    n_frames = spectrogram.shape[0]
    input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[input_])) for input_ in seq]
    mel_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in mel_spectrogram]
    spec_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in spectrogram]
    wav_feature = [tf.train.Feature(float_list=tf.train.FloatList(value=[sample])) for sample in wav]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'mel': tf.train.FeatureList(feature=mel_features),
        'spec': tf.train.FeatureList(feature=spec_features),
        'wav': tf.train.FeatureList(feature=wav_feature),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    n_frame_ = tf.train.Feature(int64_list=tf.train.Int64List(value=[n_frames]))
    input_lengths_ = tf.train.Feature(int64_list=tf.train.Int64List(value=[input_lengths]))
    identity_ = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(id)]))
    context = tf.train.Features(feature={
        "n_frame": n_frame_,
        "input_lengths": input_lengths_,
        "identity":identity_
    })
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return example



def write_worker(q_out, tfrecord_file):
    pre_time = time.time()
    count = 1
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    while True:
        deq = q_out.get()
        if deq is None:
            break
        serial = deq
        writer.write(serial.SerializeToString())
        if count % 100 == 0:
            cur_time = time.time()
            print('time:', cur_time - pre_time, 'count:', count)
            pre_time = cur_time
        count += 1


def audio_encoder(item, q_out):
    wav_root = item[0]
    content = item[1]
    seq = _symbols_to_sequence(content)
    seq = np.asarray(seq)
    id = item[2]
    example = _process_utterance(wav_root, seq, id)
    q_out.put(example)


def read_worker(q_in, q_out):
    while True:
        item = q_in.get()
        if item is None:
            break
        audio_encoder(item, q_out)

def wav_to_tfrecord_read_from_text(args, text_path, data_name, id_num):
    tfrecord_dir = os.path.join(args.output, "tfrecord_tacotron_" + data_name)
    os.makedirs(tfrecord_dir, exist_ok=True)
    tfrecord_file = os.path.join(tfrecord_dir, 'tfrecord_tacotron_' + data_name +
                                 '_id_num_' + str(id_num) + '.tfrecord')

    q_in = [multiprocessing.Queue(1024) for i in range(args.num_workers)]  # num_thread  default = 32
    q_out = multiprocessing.Queue(1024)
    read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) for i in range(args.num_workers)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, tfrecord_file))
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
        with open('./train_data_dict.json', 'r') as f:
            train_data_dict = json.load(f)
    except:
        train_data_dict = {}
    train_data_dict[data_name] = tfrecord_file
    with open('./train_data_dict.json', 'w') as f:
        json.dump(train_data_dict, f)













