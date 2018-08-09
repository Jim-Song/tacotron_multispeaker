import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback


from datasets.datafeeder_tfrecord import DataFeeder
#from datasets.datafeeder2 import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text, sequence_to_text2
from util import audio, infolog, plot, ValueWindow, align

log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  log('Checkpoint path: %s' % checkpoint_path)
  #log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  if args.batch_size:
    hparams.batch_size = args.batch_size
  if args.outputs_per_step:
    hparams.outputs_per_step = args.outputs_per_step
  hparams.chinese_symbol = args.chinese_symbol
  hparams.reverse_input = args.reverse_input
  if args.chinese_symbol:
    sequence_to_text = sequence_to_text2
  hparams.bucket_len = args.bucket_len
  hparams.eos = args.eos
  hparams.embedding_channels = args.embedding_channels



  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Multi-GPU settings
    GPUs_id = eval(args.GPUs_id)
    num_GPU = len(GPUs_id)
    hparams.num_GPU = num_GPU
    models = []

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    '''
    with tf.variable_scope('datafeeder') as scope:

      if not args.input_path:
        input_path = []
        for item in os.listdir('./training'):
          if os.path.exists(os.path.join('./training', item, 'train.txt')):
            print(os.path.join('./training', item, 'train.txt'))
            input_path.append(os.path.join('./training', item, 'train.txt'))
        feeder = DataFeeder(coord, input_path, hparams)
        inputs = feeder.inputs
        input_lengths = feeder.input_lengths
        wav_target = feeder.wav
        mel_targets = feeder.mel_targets
        linear_targets = feeder.linear_targets
      else:
        input_path = [args.input_path]
        feeder = DataFeeder(coord, input_path, hparams)
        inputs = feeder.inputs
        input_lengths = feeder.input_lengths
        wav_target = feeder.wav
        mel_targets = feeder.mel_targets
        linear_targets = feeder.linear_targets
      n_frame = tf.constant([1])
    '''

    #file_list = ['./tfrecord_data2/tfrecord_all_novel_data_41.tfrecords']
    #'''
    file_list = [os.path.join('/hdd4/tfrecord_of_allnoveldata_for_tts', file) for file in os.listdir('/hdd4/tfrecord_of_allnoveldata_for_tts')]

    feeder = DataFeeder(hparams, file_list, args.saparator_between_characters)
    inputs, input_lengths, linear_targets, mel_targets, n_frame, wav = feeder._get_batch_input()
    #'''



    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('model') as scope:
      for i, GPU_id in enumerate(GPUs_id):
        with tf.device('/gpu:%d' % GPU_id):
          with tf.name_scope('GPU_%d' % GPU_id):

            models.append(None)
            models[i] = create_model(args.model, hparams)
            models[i].initialize(inputs=inputs, input_lengths=input_lengths,
                                 mel_targets=mel_targets, linear_targets=linear_targets)
            models[i].add_loss()
            models[i].add_optimizer(global_step)

            stats = add_stats(models[i])

            #tf.get_variable_scope().reuse_variables()
            print(tf.get_variable_scope())
            




    # Bookkeeping:
    step = 0
    time_window = ValueWindow(250)
    loss_window = ValueWindow(1000)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=8)

    # Train!
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      try:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        if args.restore_step:
          # Restore from a checkpoint if the user requested it.
          restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
          saver.restore(sess, restore_path)
          log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
        else:
          log('Starting new training run at commit: %s' % commit, slack=True)

        tf.train.start_queue_runners(sess=sess)

        #feeder.start_in_session(sess)
        while not coord.should_stop():


          start_time = time.time()
          loss_w = None
          """
          step, input, mel_spec, linear_outputs = sess.run([global_step, inputs, mel_targets, linear_targets])
          print('----------------------------------------------------------------------------------')
          print(input)
          print(input.shape)
          print(mel_spec)
          print(mel_spec.shape)
          print(linear_outputs)
          print(linear_outputs.shape)

          """
          #"""
          step, loss, opt, inputs2 = sess.run([global_step, models[0].loss, models[0].optimize, inputs])
          #for item in inputs2:
          #  print(item)
          #  log('Input: %s' % sequence_to_text(item))
          #print(framess)
          time_window.append(time.time() - start_time)
          loss_window.append(loss)
          message = 'Step %-7d [%.03f avg_sec/step,  loss=%.05f, avg_loss=%.05f, lossw=%.05f]' % (
            step, time_window.average,  loss, loss_window.average, loss_w if loss_w else loss)
          log(message, slack=(step % args.checkpoint_interval == 0))

          # if the gradient seems to explode, then restore to the previous step
          if loss > 2 * loss_window.average or math.isnan(loss):
            log('recover to the previous checkpoint')
            restore_step = int((step - 10) / args.checkpoint_interval) * args.checkpoint_interval
            restore_path = '%s-%d' % (checkpoint_path, restore_step)
            saver.restore(sess, restore_path)
            continue

          #if loss > 100 or math.isnan(loss):
          #  log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          #  raise Exception('Loss Exploded')

          if step % args.summary_interval == 0:
            log('Writing summary at step: %d' % step)
            summary_writer.add_summary(sess.run(stats), step)

          if step % args.checkpoint_interval == 0:

            crrt_dir = os.path.join(log_dir, str(step))
            os.makedirs(crrt_dir, exist_ok=True)

            log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
            saver.save(sess, checkpoint_path, global_step=step)
            log('Saving audio and alignment...')
            input_seq, spectrogram, alignment, wav_original, melspectogram, spec_original, mel_original = sess.run([
              models[0].inputs[0], models[0].linear_outputs[0], models[0].alignments[0], wav[0],
              models[0].mel_outputs[0], linear_targets[0], mel_targets[0]])
            waveform = audio.inv_spectrogram(spectrogram.T)
            print(waveform)
            print(wav_original)
            audio.save_wav(waveform, os.path.join(crrt_dir, 'step-%d-audio.wav' % step))
            audio.save_wav(wav_original, os.path.join(crrt_dir, 'step-%d-audio-original.wav' % step))
            np.save(os.path.join(crrt_dir, 'spec.npy'), spectrogram, allow_pickle=False)
            np.save(os.path.join(crrt_dir, 'melspectogram.npy'), melspectogram, allow_pickle=False)
            np.save(os.path.join(crrt_dir, 'spec_original.npy'), spec_original, allow_pickle=False)
            np.save(os.path.join(crrt_dir, 'mel_original.npy'), mel_original, allow_pickle=False)
            #audio.save_wav(wav, os.path.join(crrt_dir, 'step-%d-audio2.wav' % step))
            plot.plot_alignment(alignment, os.path.join(crrt_dir, 'step-%d-align.png' % step),
              info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))



            transition_params = []
            for i in range(alignment.shape[0]):
              transition_params.append([])
              for j in range(alignment.shape[0]):
                if i == j or j-i == 1:
                  transition_params[-1].append(500)
                else:
                  transition_params[-1].append(0.0)
            alignment[0][0] = 100000
            alignment2 = np.argmax(alignment, axis=0)
            alignment3 = tf.contrib.crf.viterbi_decode(alignment.T, transition_params)
            alignment4 = np.zeros(alignment.shape)
            for i, item in enumerate(alignment3[0]):
              alignment4[item, i] = 1
            plot.plot_alignment(alignment4, os.path.join(crrt_dir, 'step-%d-align2.png' % step),
                                info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))

            '''
            crrt = 0
            sample_crrt = 0
            sample_last = 0
            for i, item in enumerate(alignment3[0]):
              if item == crrt:
                sample_crrt += hparams.sample_rate * hparams.frame_shift_ms * 6 / 1000
              if not item == crrt:
                crrt += 1
                print('item')
                print(item)
                sample_crrt = int(sample_crrt)
                sample_last = int(sample_last)
                wav_crrt = waveform[ : sample_crrt]
                wav_crrt2 = waveform[sample_last : sample_crrt]
                print('i')
                print(i)
                audio.save_wav(wav_crrt, os.path.join(crrt_dir, '%d.wav' % crrt))
                audio.save_wav(wav_crrt2, os.path.join(crrt_dir, '%d-2.wav' % crrt))
                sample_last = sample_crrt
                sample_crrt += hparams.sample_rate * hparams.frame_shift_ms * 6 / 1000
                print('wav_crrt_len')
                print(len(wav_crrt))
  
  
            print('wavform_len')
            print(len(waveform))
            '''


            #print('alignment.shape: %s' % str(alignment.shape))
            #print('alignment: %s' % alignment)
            #print(model_tacotron.alignments[0])
            #print(model_tacotron.alignments[0].shape)
            #print('alignment2: %s' % alignment2)
            #print('alignment3: %s' % alignment3[0])
            print('length alignment2')
            print(alignment2.size)
            print('input_seq.shape: %s' % str(input_seq.shape))
            print('spectrogram.shape: %s' % str(spectrogram.shape))
            print('wav_target.shape: %s' % str(wav.shape))

            input_seq2 = []
            input_seq3 = []
            for item in alignment2:
              input_seq2.append(input_seq[item])
            for item in alignment3[0]:
              input_seq3.append(input_seq[item])

            #output
            path_align1 = os.path.join(crrt_dir, 'step-%d-align1.txt' % step)
            path_align2 = os.path.join(crrt_dir, 'step-%d-align2.txt' % step)
            path_align3 = os.path.join(crrt_dir, 'step-%d-align3.txt' % step)
            path_seq1 = os.path.join(crrt_dir, 'step-%d-input1.txt' % step)
            path_seq2 = os.path.join(crrt_dir, 'step-%d-input2.txt' % step)
            path_seq3 = os.path.join(crrt_dir, 'step-%d-input3.txt' % step)
            with open(path_align1, 'w') as f:
              for row in alignment:
                for item in row:
                  f.write('%.3f' % item)
                  f.write('\t')
                f.write('\n')
            with open(path_align2, 'w') as f:
              for item in alignment2:
                f.write('%.3f' % item)
                f.write('\t')
            with open(path_align3, 'w') as f:
              for item in alignment3[0]:
                f.write('%.3f' % item)
                f.write('\t')
            with open(path_seq1, 'w') as f:
              f.write(sequence_to_text(input_seq))
            with open(path_seq2, 'w') as f:
              f.write(sequence_to_text(input_seq2))
            with open(path_seq3, 'w') as f:
              f.write(sequence_to_text(input_seq3))

            #print('input_seq.shape: %s' % input_seq.shape)
            log('Input: %s' % sequence_to_text(input_seq))
            #log('Input2: %s' % sequence_to_text(input_seq2))
            #log('Input3: %s' % sequence_to_text(input_seq3))
            #time.sleep(200)
            #'''
            #"""


        #train_threads = []
        #for model in models:
        #  train_threads.append(threading.Thread(target=train_func, args=(global_step, model, time_window, loss_window, sess, saver,)))
        #for t in train_threads:
        #  t.start()
        #for t in train_threads:
        #  t.join()

      except Exception as e:
        log('Exiting due to exception: %s' % e, slack=True)
        traceback.print_exc()
        coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./logs/')
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--GPUs_id', default='[0]', help='The GPUs\' id list that will be used. Default is 0')
  parser.add_argument('--preprocess_thread', type=int, default=1, help='preprocess_thread.')
  parser.add_argument('--description', default=None, help='description of the model')
  parser.add_argument('--batch_size', default=None, type=int, help='batch size')
  parser.add_argument('--outputs_per_step', default=None, type=int, help='outputs_per_step could be 2 3 4 6 12')
  parser.add_argument('--chinese_symbol', default=None, type=int, help='the path store the list of symbols')
  parser.add_argument('--input_path', default=None, help='the corpus')
  parser.add_argument('--reverse_input', type=int, default=0, help='reverse_input')
  parser.add_argument('--bucket_len', type=int, default=1, help='bucket_len')
  parser.add_argument('--eos', type=int, default=1, help='whether ues eos in the input sequence')
  parser.add_argument('--initial_learning_rate', type=float, default=None, help='initial_learning_rate')
  parser.add_argument('--embedding_channels', type=int, default=256, help='embedding_channels')
  parser.add_argument('--saparator_between_characters', type=bool, default=None,
                      help='insert the symbol _ between characters')





  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s-%s' % (run_name, args.description))
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
