import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence, text_to_sequence2
from util.infolog import log


_batches_per_group = 25
_p_cmudict = 0.5
_pad = 0


class DataFeeder(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, metadata_filename, hparams):
        super(DataFeeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._offset = 0
        self.metadata_filename = metadata_filename
        self.chinese_symbol = hparams.chinese_symbol
        # Load metadata:
        #self._datadir = os.path.dirname(metadata_filename[0])
        self._metadata = []
        self.time_all = 0
        '''
        for item in self.metadata_filename:
            with open(item, 'r') as f:
                metadata_tmp = [line.strip().split('|') for line in f]
            self._datadir = os.path.dirname(item)
            metadata_tmp2 = []
            for item in metadata_tmp:
                metadata_tmp2.append([os.path.join(self._datadir, item[0]),#item[0],#os.path.join(self._datadir, item[0]),
                                      os.path.join(self._datadir, item[1]),#item[1],#os.path.join(self._datadir, item[1]),
                                      item[2],
                                      item[3],
                                      item[4]])
            self._metadata = self._metadata + metadata_tmp2
        hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
        log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))
        '''
        self._metadata=[['/home/songjinming/tts/tw_18_6_29/0房玄龄却是明白苺煖/spec2.npy','/home/songjinming/tts/tw_18_6_29/0房玄龄却是明白苺煖/melspectogram2.npy',[],'房玄龄却是明白',[]],
                        ['/home/songjinming/tts/tw_18_6_29/0小智深深的吸了一口气佉煖/spec2.npy','/home/songjinming/tts/tw_18_6_29/0小智深深的吸了一口气佉煖/melspectogram2.npy',[],'小智深深的吸了一口气',[]],
                        ['/home/songjinming/tts/tw_18_6_29/0沈制片的朋友就是我的朋友澙煖/spec2.npy','/home/songjinming/tts/tw_18_6_29/0沈制片的朋友就是我的朋友澙煖/melspectogram2.npy',[],'沈制片的朋友就是我的朋友',[]]]


        print('length of metadata:%d' % len(self._metadata))
        random.shuffle(self._metadata)


        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, None, 'wav'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
            ]

        # Create queue for buffering data:
        queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32], name='input_queue')

        self._enqueue_op = queue.enqueue(self._placeholders)

        #self.inputs, self.input_lengths, self.mel_targets, self.linear_targets, self.spec_lengths = queue.dequeue()
        self.inputs, self.input_lengths, self.wav, self.mel_targets, self.linear_targets = queue.dequeue()

        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.wav.set_shape(self._placeholders[2].shape)
        self.mel_targets.set_shape(self._placeholders[3].shape)
        self.linear_targets.set_shape(self._placeholders[4].shape)
        #self.spec_lengths.set_shape(self._placeholders[4].shape)


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
        n = self._hparams.batch_size# * max((self._hparams.outputs_per_step - 1), 1)
        r = self._hparams.outputs_per_step

        examples = [self._get_next_example() for i in range(n * _batches_per_group)]

       # while len(examples) < self.n * _batches_per_group:
       #     next_example = None
       #     while next_example is None:
       #         next_example = self._get_next_example()

       #     examples.append(next_example)

        examples.sort(key=lambda x: x[1])
        batches = [examples[i:i+n] for i in range(0, len(examples), n)]
        #for i in range(0, len(examples), self.n):
        #    batches.append(examples[i:i + self.n])

        # split the batches in case the big wav file induct Out Of Memory
        #def split_list(list, n):
        #    output = []
        #    for i in range(n):
        #        output.append(list[int(len(list) * i / n): int(len(list) * (i + 1) / n)])
        #    return output
        '''
            batches2 = []
            for i, batch in enumerate(batches):
                print('len of spec %d' % batches[i][-1][1])
                if batches[i][-1][1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 16:
                    del (batches[i])
                    self.ct_toolong += 1
                    log('a too long batch is deleted %d ' % (self.ct_toolong))
                    # print(len(batches2))
                elif batches[i][-1][1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 8:
                    del (batches[i])
                    split_batch = split_list(batch, 16)
                    for item in split_batch:
                        if item:
                            batches2.append(item)
                elif batches[i][-1][1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 4:
                    del (batches[i])
                    split_batch = split_list(batch, 8)
                    for item in split_batch:
                        if item:
                            batches2.append(item)
                elif batches[i][-1][1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 2:
                    del (batches[i])
                    split_batch = split_list(batch, 4)
                    for item in split_batch:
                        if item:
                            batches2.append(item)
                        # print(len(batches2))
                elif batches[i][-1][1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 1:
                    del (batches[i])
                    split_batch = split_list(batch, 2)
                    for item in split_batch:
                        if item:
                            batches2.append(item)
                        # print(len(batches2))
                else:
                    batches2.append(batch)
                    # print(len(batches2))
            batches = batches2
        '''
        self.time_all += time.time() - start
        log('1Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        print(self._offset)

        for batch in batches:
            input_batch = _prepare_batch(batch, r)
            '''
            print('*****************************************************************************************************')
            print('batch:')
            print(input_batch)

            for item in input_batch:
                print('------------------------------------------------------------------')
                print(item)
                print('------------------------------------------------------------------')
            print('*****************************************************************************************************')
            '''
            '''
            batch_size = len(input_batch[0]) / self._hparams.num_GPU
            print('1the num of data on a single GPU is %f' % batch_size)
            batch_size = len(input_batch[1]) / self._hparams.num_GPU
            print('2the num of data on a single GPU is %f' % batch_size)
            batch_size = len(input_batch[2]) / self._hparams.num_GPU
            print('3the num of data on a single GPU is %f' % batch_size)
            batch_size = len(input_batch[3]) / self._hparams.num_GPU
            print('4the num of data on a single GPU is %f' % batch_size)

            batch_size = input_batch[0].shape
            print('the shape of input is ' + str(batch_size))
            print('the shape of input is ' + str(type(input_batch[0][0][0])))
            batch_size = input_batch[1].shape
            print('the shape of input is ' + str(batch_size))
            print('the shape of input is ' + str(type(input_batch[1][0])))
            batch_size = input_batch[2].shape
            print('the shape of input is ' + str(batch_size))
            print('the shape of input is ' + str(type(input_batch[2][0][0][0])))
            batch_size = input_batch[3].shape
            print('the shape of input is ' + str(batch_size))
            print('the shape of input is ' + str(type(input_batch[3][0][0][0])))
            '''
            feed_dict = dict(zip(self._placeholders, input_batch))

            self._session.run(self._enqueue_op, feed_dict=feed_dict)



    def _get_next_example(self):


        #if self._offset % 1000 == 0:
        #    print('___________________________________________________________________________________________________')
        #    print('time of 1000:' + str(time.time() - self.start_time))
        #    #print('___________________________________________________________________________________________________')
        #    self.start_time = time.time()

        #metas = []

        #print('_get_next_example')
        '''
        for i in range( int(self.n * _batches_per_group + self.n) ):
            if self._offset >= len(self._metadata):
                self._offset = 0
                random.shuffle(self._metadata)
            meta = self._metadata[self._offset]
            self._offset += 1
            metas.append(meta)
        #outputs = []
        '''

        if self._offset >= len(self._metadata):
            self._offset = 0
            random.shuffle(self._metadata)
            print('shuffle')
            print(self.time_all)
            self.time_all = 0
        meta = self._metadata[self._offset]
        self._offset += 1


        text = meta[3]
        #if self._hparams.reverse_input:
        #    text = list(text)
        #    text.reverse()
        #    ''.join(text)



        #av_path = meta[4]
        #spectrogram_filename = meta[0]
        #mel_filename = meta[1]

        #start = time.time()
        linear_target = np.load(meta[0])
        mel_target = np.load(meta[1])
        #print(time.time() - start)
        #print(os.path.join(self._datadir, meta[0]))

        if self.chinese_symbol:
            input_data = np.asarray(text_to_sequence2(text, ['english_cleaners']), dtype=np.int32)
        else:
            input_data = np.asarray(text_to_sequence(text, ['english_cleaners']), dtype=np.int32)
        len_input_data = len(input_data)
        wav = np.array([1, 2, 3, 4, 5])
        #(input_data, len_input_data, wav, mel_target, linear_target) = \
        #    preprocess_data(text, wav_path, self.chinese_symbol, spectrogram_filename, mel_filename)

            #executor = ProcessPoolExecutor(max_workers=14)
            #outputs.append(executor.submit(partial(preprocess_data, text, wav_path, self.chinese_symbol, spectrogram_filename, mel_filename)))
        #input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
        #linear_target, mel_target, n_frames= get_linear_and_mel_targert(wav_path=meta[0])
        #print(meta[0] + ':' + str(time.time()-start_time))
        #return (input_data, mel_target, linear_target, len(linear_target))
        #try:#this error occurs constantly and it is better to use try here
            #Error:concurrent.futures.process.BrokenProcessPool: A process in the process pool was
            #      terminated abruptly while the future was running or pending.
        #final_output = [output.result() for output in outputs]
        #except:
        #    final_output = [None,]
        #return final_output

        return (input_data, len_input_data, wav, mel_target, linear_target)





    '''
    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
            time.sleep(15)
        return self.threads
        '''



def preprocess_data(text, wav_path, chinese_symbol=False, spectrogram_filename=None, mel_filename=None):
    if not spectrogram_filename:
        wav, linear_target, mel_target, n_frames = get_wav_linear_and_mel_targert(wav_path)
    else:
        #linear_target = np.load(os.path.join('./training', spectrogram_filename))
        #mel_target = np.load(os.path.join('./training', mel_filename))
        linear_target = np.load(spectrogram_filename)
        mel_target = np.load(mel_filename)
        #wav = audio.load_wav(wav_path)
        wav = np.array([1, 2, 3, 4, 5])
    #if chinese_symbol:
    #    input_data = np.asarray(text_to_sequence2(text, ['english_cleaners']), dtype=np.int32)
    #else:
        input_data = np.asarray(text_to_sequence(text, ['english_cleaners']), dtype=np.int32)
    #if n_frames < 1000:
    #return (input_data, len(input_data) * n_frames, wav, mel_target, linear_target)
    return (input_data, len(input_data), wav, mel_target, linear_target)
    #else:
    #    return None



def get_wav_linear_and_mel_targert(wav_path, set_spec_length=None):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # Return a tuple describing this training example:
    if set_spec_length is not None:
        return (spectrogram.T[:set_spec_length], mel_spectrogram.T[:set_spec_length], n_frames)
    #wav = wav.reshape(-1, 1)
    #wav = np.pad(wav, [[2048, 0], [0, 0]], 'constant')
    #wav = np.pad(wav, [[2048, 0]], 'constant')
    return (wav, spectrogram.T, mel_spectrogram.T, n_frames)





def _prepare_batch(batch, outputs_per_step, set_spec_length=None):
    '''
    :param batch: [(input_data, len(linear_target), wav, mel_target, linear_target) * batch_size]
    :param outputs_per_step:
    :param set_spec_length:
    :return:
    '''
    #random.shuffle(batch)
    inputs = _prepare_inputs([x[0] for x in batch])
    # ('inputs'+str(inputs.shape))
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
    # print('input_lengths' + str(input_lengths))
    wav = _prepare_inputs([x[2] for x in batch])
    #wav = np.asarray([[1,2,3,4,5],[6,7,8,9,0]])
    mel_targets = _prepare_targets([x[3] for x in batch], outputs_per_step, set_spec_length)
    # print('mel_targets' + str(mel_targets))
    linear_targets = _prepare_targets([x[4] for x in batch], outputs_per_step, set_spec_length)
    #spec_lengths = np.asarray([linear_target.shape[0] for linear_target in linear_targets], dtype=np.int32)
    #print('0:' + str(linear_targets[0].shape[0]))
    #print('1:' + str(linear_targets[0].shape[1]))
    #print('2:' + str(linear_targets[0].shape[2]))
    #print('0:' + str(linear_targets[1].shape[0]))
    #print('1:' + str(linear_targets[1].shape[1]))
    #print('2:' + str(linear_targets[1].shape[2]))
    return (inputs, input_lengths, wav, mel_targets, linear_targets)

def _prepare_inputs(inputs, ):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment, set_spec_length):
  max_len = max((len(t) for t in targets)) + 1
  if set_spec_length is not None:
      max_len = set_spec_length
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
















"""
import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log


_batches_per_group = 32
_p_cmudict = 0.5
_pad = 0


class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, metadata_filename, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    self._offset = 0

    # Load metadata:
    self._datadir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
      hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
      log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
      tf.placeholder(tf.int32, [None, None], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)

    # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
    # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
    # synthesis (useful for proper nouns, etc.)
    if hparams.use_cmudict:
      cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b')
      if not os.path.isfile(cmudict_path):
        raise Exception('If use_cmudict=True, you must download ' +
          'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s'  % cmudict_path)
      self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
      log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
    else:
      self._cmudict = None


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
    examples.sort(key=lambda x: x[-1])
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
    self._offset += 1

    text = meta[3]
    if self._cmudict and random.random() < _p_cmudict:
      text = ' '.join([self._maybe_get_arpabet(word) for word in text.split(' ')])

    input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
    linear_target = np.load(os.path.join(self._datadir, meta[0]))
    mel_target = np.load(os.path.join(self._datadir, meta[1]))
    return (input_data, mel_target, linear_target, len(linear_target))


  def _maybe_get_arpabet(self, word):
    arpabet = self._cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
  return (inputs, input_lengths, mel_targets, linear_targets)


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
"""