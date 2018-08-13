import argparse
from datetime import datetime
import math
import numpy as np
import os, re, json
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
    subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])     # Verify client is clean
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
    log('Using model: %s' % args.model)
    log(hparams_debug_string())

    sequence_to_text = sequence_to_text2
    hparams.bucket_len = args.bucket_len
    hparams.eos = args.eos


    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Multi-GPU settings
        GPUs_id = eval(args.GPUs_id)
        num_GPU = len(GPUs_id)
        hparams.num_GPU = num_GPU
        models = []

        # Set up DataFeeder:
        coord = tf.train.Coordinator()

        with open('./train_data_dict.json', 'r') as f:
            train_data_dict = json.load(f)

        train_data = args.train_data.split(',')
        file_list = []
        pattern = '[.]*\\_id\\_num\\_([0-9]+)[.]+'
        id_num = 0
        for item in train_data:
            file_list.append(train_data_dict[item])
            id_num += int( re.findall(pattern, train_data_dict[item])[0] )
        log('train data:%s' % args.train_data)

        feeder = DataFeeder(hparams, file_list)
        inputs, input_lengths, linear_targets, mel_targets, n_frame, wav, identity = feeder._get_batch_input()

        print('identity')
        print(identity)

        # Set up model:
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('model') as scope:
            for i, GPU_id in enumerate(GPUs_id):
                with tf.device('/gpu:%d' % GPU_id):
                    with tf.name_scope('GPU_%d' % GPU_id):

                        models.append(None)
                        models[i] = create_model(args.model, hparams)
                        models[i].initialize(inputs=inputs, input_lengths=input_lengths, mel_targets=mel_targets,
                                             linear_targets=linear_targets, identity=identity, id_num=id_num)
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
                start_queue(sess=sess)

                while not coord.should_stop():
                    start_time = time.time()
                    loss_w = None

                    step, loss, opt, inputs2 = sess.run([global_step, models[0].loss, models[0].optimize, inputs])

                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    message = 'Step %-7d [%.03f avg_sec/step,    loss=%.05f, avg_loss=%.05f, lossw=%.05f]' % (
                        step, time_window.average,    loss, loss_window.average, loss_w if loss_w else loss)
                    log(message, slack=(step % args.checkpoint_interval == 0))

                    # if the gradient seems to explode, then restore to the previous step
                    if loss > 2 * loss_window.average or math.isnan(loss):
                        log('recover to the previous checkpoint')
                        restore_step = int((step - 10) / args.checkpoint_interval) * args.checkpoint_interval
                        restore_path = '%s-%d' % (checkpoint_path, restore_step)
                        saver.restore(sess, restore_path)
                        continue

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
                        audio.save_wav(waveform, os.path.join(crrt_dir, 'step-%d-audio.wav' % step))
                        audio.save_wav(wav_original, os.path.join(crrt_dir, 'step-%d-audio-original.wav' % step))
                        np.save(os.path.join(crrt_dir, 'spec.npy'), spectrogram, allow_pickle=False)
                        np.save(os.path.join(crrt_dir, 'melspectogram.npy'), melspectogram, allow_pickle=False)
                        np.save(os.path.join(crrt_dir, 'spec_original.npy'), spec_original, allow_pickle=False)
                        np.save(os.path.join(crrt_dir, 'mel_original.npy'), mel_original, allow_pickle=False)
                        plot.plot_alignment(alignment, os.path.join(crrt_dir, 'step-%d-align.png' % step),
                            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
                        #提取alignment， 看看对其效果如何
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

                        crrt = 0
                        sample_crrt = 0
                        sample_last = 0
                        for i, item in enumerate(alignment3[0]):
                            if item == crrt:
                                sample_crrt += hparams.sample_rate * hparams.frame_shift_ms * hparams.outputs_per_step / 1000
                            if not item == crrt:
                                crrt += 1
                                sample_crrt = int(sample_crrt)
                                sample_last = int(sample_last)
                                wav_crrt = waveform[ : sample_crrt]
                                wav_crrt2 = waveform[sample_last : sample_crrt]
                                audio.save_wav(wav_crrt, os.path.join(crrt_dir, '%d.wav' % crrt))
                                audio.save_wav(wav_crrt2, os.path.join(crrt_dir, '%d-2.wav' % crrt))
                                sample_last = sample_crrt
                                sample_crrt += hparams.sample_rate * hparams.frame_shift_ms * hparams.outputs_per_step / 1000

                        input_seq2 = []
                        input_seq3 = []
                        for item in alignment2:
                            input_seq2.append(input_seq[item])
                        for item in alignment3[0]:
                            input_seq3.append(input_seq[item])

                        #output alignment
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

                        log('Input: %s' % sequence_to_text(input_seq))

            except Exception as e:
                #log('Exiting due to exception: %s' % e, slack=True)
                #traceback.print_exc()
                coord.request_stop(e)

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./logs/')
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
    parser.add_argument('--bucket_len', type=int, default=1, help='bucket_len')
    parser.add_argument('--eos', type=_str_to_bool, default='True', help='whether ues eos in the input sequence')
    parser.add_argument('--initial_learning_rate', type=float, default=None, help='initial_learning_rate')
    parser.add_argument('--train_data', type=str, default='THCHS', help='training datas to be used')




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
