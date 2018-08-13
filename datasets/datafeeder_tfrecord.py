#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os, random, re
import numpy as np



class DataFeeder():

    def __init__(self, hparams, file_list):
        """构造从TFRecord读取数据的图.

        Args:
            hparams: 参数
            file_list: record文件列表.
        """
        self.hp = hparams
        self.file_list = file_list

        pattern = '[.]*\\_id\\_num\\_([0-9]+)[.]+'
        id_num = 0

        with tf.device("/cpu:0"):
            self.enqueue = []
            self.num_people = 0
            self.queues = []
            self.queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32], name='input_queue')
            for i, file in enumerate(self.file_list):
                print(file)
                filename_queue = tf.train.string_input_producer([file])
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)

                #bucket_boundaries = list(range(80, self.hp.max_iters*self.hp.outputs_per_step, self.hp.outputs_per_step * 5))
                bucket_boundaries = list(range(0, 50, self.hp.bucket_len))
                queue_capacity = (len(bucket_boundaries) + 1) * self.hp.batch_size

                self.queues.append(tf.RandomShuffleQueue(queue_capacity, 100, tf.string))

                input_example = self.queues[i].enqueue(serialized_example)
                tf.train.add_queue_runner(tf.train.QueueRunner(self.queues[i], [input_example]))

                context, feat_list = tf.parse_single_sequence_example(
                    self.queues[i].dequeue(),
                    {
                    'input_lengths': tf.FixedLenFeature([], dtype=tf.int64),
                    'n_frame': tf.FixedLenFeature([], tf.int64),
                    'identity': tf.FixedLenFeature([], tf.int64),
                    },
                    {
                    "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    'spec': tf.FixedLenSequenceFeature([self.hp.num_freq], tf.float32),
                    'mel': tf.FixedLenSequenceFeature([self.hp.num_mels], tf.float32),
                    'wav': tf.FixedLenSequenceFeature([], tf.float32),
                    }
                )

                inputs = tf.to_int32(feat_list["inputs"])
                n_frame = tf.to_int32(context["n_frame"])
                identity = tf.to_int32(context["identity"])
                input_lengths = tf.to_int32(context['input_lengths'])
                linear_targets = feat_list['spec']#tf.decode_raw(features['spec'], tf.float32)
                mel_targets = feat_list['mel']#tf.decode_raw(features['mel'], tf.float32)
                wav = feat_list['wav']#tf.decode_raw(features['wav'], tf.float32)
                n_frame = tf.shape(inputs)[0]

                if self.hp.eos:
                    eos_crrt = tf.random_uniform([1], minval=7000, maxval=13550, dtype=tf.int32)
                    inputs = tf.concat([inputs, eos_crrt], 0)

                _, __batch_tensors = tf.contrib.training.bucket_by_sequence_length(n_frame,
                    [inputs, input_lengths, linear_targets, mel_targets, n_frame, wav, identity + tf.constant(id_num)],
                    self.hp.batch_size, bucket_boundaries, capacity=30, dynamic_pad=True, num_threads=25)
                self.enqueue.append(self.queue.enqueue(__batch_tensors))

                id_num += int(re.findall(pattern, file)[0])

            self.inputs, self.input_lengths, self.linear_targets, self.mel_targets, self.n_frame, self.wav, \
            self.identity = self.queue.dequeue()


    def _get_batch_input(self):
        """返回一组训练数据输入.

        Returns:
          6元组, 包括帧数, 标签, 特征 etc.
        """
        inputs, input_lengths, linear_targets, mel_targets, n_frame, wav, identity = \
            tuple([self.inputs, self.input_lengths, self.linear_targets, self.mel_targets,
                  self.n_frame, self.wav, self.identity])
        return inputs, input_lengths, linear_targets, mel_targets, n_frame, wav, identity


    def start_queue(self, sess):
        while True:
            i = randint(0, len(self.file_list)-1)
            sess.run(self.enqueue[i])


def _shuffle_inputs(input_tensors, capacity, min_after_dequeue, num_threads):
    """Shuffles tensors in `input_tensors`, maintaining grouping."""
    shuffle_queue = tf.RandomShuffleQueue(
        capacity, min_after_dequeue, dtypes=[t.dtype for t in input_tensors])
    enqueue_op = shuffle_queue.enqueue(input_tensors)
    runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
    tf.train.add_queue_runner(runner)

    output_tensors = shuffle_queue.dequeue()

    for i in range(len(input_tensors)):
        output_tensors[i].set_shape(input_tensors[i].shape)

    return output_tensors



def count_records(file_list, stop_at=None):
    """Counts number of records in files from `file_list` up to `stop_at`.
    Args:
      file_list: List of TFRecord files to count records in.
      stop_at: Optional number of records to stop counting at.
    Returns:
      Integer number of records in files from `file_list` up to `stop_at`.
    """
    num_records = 0
    for tfrecord_file in file_list:
        tf.logging.info('Counting records in %s.', tfrecord_file)
        print('Counting records in %s.', tfrecord_file)
        for _ in tf.python_io.tf_record_iterator(tfrecord_file):
            num_records += 1
            if stop_at and num_records >= stop_at:
                print('Number of records is at least %d.', num_records)
                tf.logging.info('Number of records is at least %d.', num_records)
                return num_records
    tf.logging.info('Total records: %d', num_records)
    print('Total records: %d', num_records)
    return num_records


















