#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os, random, re, threading
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
            self.queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32],
                                      name='input_queue')
            for i, file in enumerate(self.file_list):
                print(file)
                filename_queue = tf.train.string_input_producer([file])
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)
                bucket_boundaries = list(range(0, 50, self.hp.bucket_len))
                queue_capacity = (len(bucket_boundaries) + 1) * self.hp.batch_size
                self.queues.append(tf.FIFOQueue(queue_capacity, tf.string))
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

            self.inputs.set_shape([self.hp.batch_size, None])
            self.input_lengths.set_shape([self.hp.batch_size])
            self.mel_targets.set_shape([self.hp.batch_size, None, hparams.num_mels])
            self.linear_targets.set_shape([self.hp.batch_size, None, hparams.num_freq])
            self.n_frame.set_shape([self.hp.batch_size])
            self.wav.set_shape([self.hp.batch_size, None])
            self.identity.set_shape([self.hp.batch_size])


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
        while not self.coord.should_stop():
            i = random.randint(0, len(self.file_list)-1)
            sess.run(self.enqueue[i])


    def start_threads(self, sess, coord, n_threads=1):
        self.threads = []
        self.coord = coord
        for _ in range(n_threads):
            thread = threading.Thread(target=self.start_queue, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

