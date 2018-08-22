import numpy as np
import os, json
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence, text_to_sequence2
from util.infolog import log


_batches_per_group = 20
_pad = 0


class DataFeeder(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, hparams, file_list, coordinator):
        super(DataFeeder, self).__init__()
        self._hparams = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._offset = 0
        self._metadata = []
        self._coord = coordinator
        self._p_phone_sub = 0.5

        # Load metadata:
        id_num = 0
        for file in file_list:
            with open(file, encoding='utf-8') as f:
                id_num_crrt = 0
                crrt_metadata = []
                for line in f:
                    line = eval(line)
                    if line[4] > id_num_crrt:
                        id_num_crrt = line[4]
                    line[4] = line[4] + id_num
                    crrt_metadata.append(line)
                id_num += id_num_crrt + 1
                self._metadata = self._metadata + crrt_metadata
                log('No. of samples from %s is %d' % (file, len(crrt_metadata)))
        random.shuffle(self._metadata)


        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
            tf.placeholder(tf.float32, [None, None], 'wavs'),
            tf.placeholder(tf.int32, [None], 'identities'),
        ]

        # Create queue for buffering data:
        queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32], name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.inputs, self.input_lengths, self.mel_targets, self.linear_targets, self.wavs, self.identities = queue.dequeue()
        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.mel_targets.set_shape(self._placeholders[2].shape)
        self.linear_targets.set_shape(self._placeholders[3].shape)
        self.wavs.set_shape(self._placeholders[4].shape)
        self.identities.set_shape(self._placeholders[5].shape)

        # Load phone dict: If enabled, this will randomly substitute some words in the training data with
        # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
        # synthesis (useful for proper nouns, etc.)
        if hparams.per_cen_phone_input:
            char_2_phone_dict_path = './datasets/char_2_phone_dict.json'
            if not os.path.isfile(char_2_phone_dict_path):
                raise Exception('no char_2_phone dict found')
            with open(char_2_phone_dict_path, 'r') as f:
                self._phone_dict = json.load(f)
                log('Loaded characters to phones dict from %s' % char_2_phone_dict_path)
        else:
            self._phone_dict = None


    def start_in_session(self, session):
        self._session = session
        self.start()


    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)


    def _enqueue_next_group(self):
        start = time.time()

        # Read a group of examples:
        n = self._hparams.batch_size
        r = self._hparams.outputs_per_step
        examples = [self._get_next_example() for i in range(n * _batches_per_group)]

        # Bucket examples based on similar output sequence length for efficiency:
        examples.sort(key=lambda x: x[-3])
        batches = [examples[i:i+n] for i in range(0, len(examples), n)]
        random.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)


    def _get_next_example(self):
        '''Loads a single example (input, mel_target, linear_target, cost) from disk'''

        if self._offset >= len(self._metadata):
            self._offset = 0
            random.shuffle(self._metadata)
        meta = self._metadata[self._offset]
        # meta = ['/ssd1/npy_tacotron_THCHS_id_num_60/THCHS-spec-00151.npy',
        #         '/ssd1/npy_tacotron_THCHS_id_num_60/THCHS-mel-00151.npy',
        #         '/ssd1/npy_tacotron_THCHS_id_num_60/THCHS-wav-00151.npy',
        #         '职工 们 爱 厂 爱岗 爱 产品 心 往 一处 想 劲儿 往 一处 使',
        #         25]
        self._offset += 1
        text = meta[3]
        if self._phone_dict :
            self._p_phone_sub = random.random() - 0.5 + (self._hparams.per_cen_phone_input * 2 -0.5)
            text = ''.join([self._maybe_get_arpabet(word) for word in text.split(' ')])
        input_data = np.asarray(text_to_sequence2(text, self._cleaner_names), dtype=np.int32)
        linear_target = np.load(meta[0])
        mel_target = np.load(meta[1])
        wav = np.load(meta[2])
        identity = meta[4]
        return (input_data, mel_target, linear_target, len(input_data), wav, identity)


    def _maybe_get_arpabet(self, word):
        try:
            phone = self._phone_dict[word]
            phone = ' '.join(phone)
        except:
            phone = None
            log('%s is not found in the char 2 phone dict' % word)
        return '{%s}' % phone if phone is not None and random.random() < self._p_phone_sub else word


def _prepare_batch(batch, outputs_per_step):
    random.shuffle(batch)
    inputs = _prepare_inputs([x[0] for x in batch])
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
    mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
    linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
    wavs = _prepare_inputs([x[4] for x in batch])
    identities = np.asarray([x[5] for x in batch], dtype=np.int32)
    return (inputs, input_lengths, mel_targets, linear_targets, wavs, identities)


def _prepare_inputs(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
