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

def _prepare_targets(targets, alignment):
    max_len = len(targets) + 1
    return _pad_target(targets, _round_up(max_len, alignment))

def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=0)

def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


def _process_utterance(wav_path, seq, writer):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32).T


    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T



    mel_spectrogram = _prepare_targets(mel_spectrogram, hparams.outputs_per_step*2)
    spectrogram = _prepare_targets(spectrogram, hparams.outputs_per_step*2)


    '''
    spec_raw = spectrogram.tostring()
    mel_raw = mel_spectrogram.tostring()
    seq_raw = seq.tostring()
    wav_raw = wav.tostring()
    '''
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
    context = tf.train.Features(feature={
        "n_frame": n_frame_,
        "input_lengths": input_lengths_,
    })

    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)






    '''
    example = tf.train.Example(features=tf.train.Features(feature={
        'inputs': _bytes_feature(seq_raw),
        'input_lengths': _int64_feature(input_lengths),
        'spec': _bytes_feature(spec_raw),
        'mel': _bytes_feature(mel_raw),
        'n_frame': _int64_feature(n_frames),
        'wav': _bytes_feature(wav_raw),
    }))
    '''


    writer.write(example.SerializeToString())

  # Return a tuple describing this training example:
    return None


def write_to_tfrecord(item, tfrecord_dir):
    #tfrecord output dir
    filename = os.path.join(tfrecord_dir, "tfrecord_all_novel_data_" + item + "_padlength" + str(hparams.outputs_per_step*2) + ".tfrecords")
    #generate a writer
    writer = tf.python_io.TFRecordWriter(filename)
    #for item in os.listdir('/home/pattern/songjinming/tts/data/all_novel_data/id_separate'):

    in_dir = os.path.join('/home/pattern/songjinming/tts/data/all_novel_data/id_separate', item)

    wav_path = os.path.join(in_dir, 'wav')
    stm_path = os.path.join(in_dir, 'stm')

    for wav_file in os.listdir(wav_path):
        name_file = os.path.splitext(wav_file)[0]
        if not os.path.splitext(wav_file)[1] == '.wav':
            continue
        txt_file = '.'.join([name_file, 'stm'])
        wav_root = os.path.join(wav_path, wav_file)
        txt_root = os.path.join(stm_path, txt_file)

        with open(txt_root, 'r') as f:
            content = f.read()

        content = _curly_re.match(content)
        content = content.group(3)
        content = content.split(' ')
        content = ''.join(content)
        seq = _symbols_to_sequence(content)
        seq = np.asarray(seq)
        #print(seq)

        #futures.append(executor.submit(partial(_process_utterance, wav_root, seq, writer)))
        tmp = _process_utterance(wav_root, seq, writer)

    #tmp = [future.result() for future in tqdm(futures)]

    writer.close()



_pad        = '_'
_eos        = '~'
_curly_re = re.compile(r'(.*?)(<.+?>)(.*)')

with open('/home/pattern/songjinming/tts/tw_tacotron_wavenet/datasets/character_list.json', 'r') as f:
    _characters = json.load(f)

symbols = [_pad, _eos] + _characters
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

raw_data_path = '/home/pattern/songjinming/tts/data/all_novel_data/id_separate'
tfrecord_dir = "../tfrecord_data/"

os.makedirs(tfrecord_dir, exist_ok=True)
executor = ProcessPoolExecutor(max_workers=30)
futures = []

dir_names = []
already_exist = os.listdir(raw_data_path)

for item in os.listdir(raw_data_path):
    if item[0] == '1':
        #continue
        futures.append(executor.submit(partial(write_to_tfrecord, item, tfrecord_dir)))

[future.result() for future in futures]






