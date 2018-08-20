import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence, text_to_sequence2, sequence_to_text2
from util import audio
from util import plot
from tensorflow.python import pywrap_tensorflow


class Synthesizer:
    def load(self, checkpoint_path, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        identity = tf.placeholder(tf.int32, [1], 'identity')
        with tf.variable_scope('model') as scope:
            hparams.chinese_symbol = True
            self.model = create_model(model_name, hparams)
            reader2 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
            var_to_shape_map = reader2.get_variable_to_shape_map()
            id_num = var_to_shape_map['model/inference/embedding_id'][0]
            self.model.initialize(inputs, input_lengths, identity=identity, id_num=id_num)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
            self.alignment = self.model.alignments[0]

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, text, identity, path=None, path_align=None):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence2(text, cleaner_names)[:-1]
        print(seq)
        print(sequence_to_text2(seq))
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
            self.model.identity: np.asarray([identity], dtype=np.int32),
        }
        wav, alignment = self.session.run([self.wav_output, self.alignment], feed_dict=feed_dict)
        if path_align is not None:
            plot.plot_alignment(alignment, path_align)
        wav = audio.inv_preemphasis(wav)
        #wav = wav[:audio.find_endpoint(wav)]
        out = io.BytesIO()
        if path is not None:
            audio.save_wav(wav, path)
        else:
            audio.save_wav(wav, './1.wav')

        return out.getvalue()
