#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np



class DataFeeder():

    def __init__(self, hparams, file_list, saparator_between_characters=None):
        """构造从TFRecord读取数据的图.

        Args:
            hparams: 参数
            file_list: record文件列表.
        """
        self.hp = hparams
        self.file_list = file_list

        with tf.device("/cpu:0"):

            filename_queue = tf.train.string_input_producer(self.file_list, shuffle=True)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            #bucket_boundaries = list(range(80, self.hp.max_iters*self.hp.outputs_per_step, self.hp.outputs_per_step * 5))
            bucket_boundaries = list(range(0, 50, self.hp.bucket_len))
            queue_capacity = (len(bucket_boundaries) + 1) * self.hp.batch_size

            queue = tf.RandomShuffleQueue(queue_capacity, 100, tf.string)

            input_example = queue.enqueue(serialized_example)
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, [input_example]))

            context, feat_list = tf.parse_single_sequence_example(
                queue.dequeue(),
                {
                'input_lengths': tf.FixedLenFeature([], dtype=tf.int64),
                'n_frame': tf.FixedLenFeature([], tf.int64),
                },
                {
                "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                'spec': tf.FixedLenSequenceFeature([self.hp.num_freq], tf.float32),
                'mel': tf.FixedLenSequenceFeature([self.hp.num_mels], tf.float32),
                'wav': tf.FixedLenSequenceFeature([], tf.float32),
                }
            )

            self.inputs = tf.to_int32(feat_list["inputs"])
            self.n_frame = tf.to_int32(context["n_frame"])
            self.input_lengths = tf.cast(context['input_lengths'], tf.int64)
            self.linear_targets = feat_list['spec']#tf.decode_raw(features['spec'], tf.float32)
            #self.linear_targets = tf.transpose(self.linear_targets)
            #self.linear_targets.set_shape([None, self.hp.num_mels])
            self.mel_targets = feat_list['mel']#tf.decode_raw(features['mel'], tf.float32)
            #mel_length = tf.shape(self.mel_targets)[0]
            #pad_len = tf.mod(5, 2)
            #pad_len = tf.add(5, 2)
            #pad_len = 10
            #paddings = tf.get_variable('asdf', initializer=[[0, pad_len], [0, 0]])
            #self.mel_targets = tf.pad(self.mel_targets,  [[0, 6], [0, 0]], "CONSTANT")
            #self.linear_targets = tf.pad(self.linear_targets, [[0, 6], [0, 0]], "CONSTANT")

            #self.mel_targets = tf.pad(self.mel_targets, [[0, 0], [0, pad_len]], "CONSTANT")
            #self.mel_targets = tf.transpose(self.mel_targets)
            #self.mel_targets.set_shape([None, self.hp.num_mels])

            self.wav = feat_list['wav']#tf.decode_raw(features['wav'], tf.float32)
            self.n_frame = tf.shape(self.inputs)[0]
            print('self.inputs')
            print(self.inputs)
            a = np.asarray([0.0, 0])
            print('a.shape')
            print(a.shape)
            if self.hp.eos:
                eos_crrt = tf.random_uniform([1], minval=7000, maxval=13550, dtype=tf.int32)
                self.inputs = tf.concat([self.inputs, eos_crrt], 0)

            if saparator_between_characters:
                z = tf.zeros(tf.shape(self.inputs), dtype=tf.int32)
                zz = tf.expand_dims(z, 1)
                print('inputs')
                print(self.inputs)
                print('zz')
                print(zz)
                print('z')
                print(z)
                self.inputs = tf.expand_dims(self.inputs, 1)
                self.inputs = tf.reshape(self.inputs, [-1, 1])
                print('inputs')
                print(self.inputs)
                e = tf.concat([zz, self.inputs], 1)
                print('e')
                print(e)
                #e = tf.transpose(e, [1,0])
                print('e2')
                print(e)
                self.inputs = tf.reshape(e, [-1])

            _, self.__batch_tensors = tf.contrib.training.bucket_by_sequence_length(self.n_frame,
                [self.inputs, self.input_lengths, self.linear_targets, self.mel_targets, self.n_frame, self.wav],
                self.hp.batch_size, bucket_boundaries, capacity=30, dynamic_pad=True, num_threads=25)

            #input_tensors = [self.inputs, self.input_lengths, self.linear_targets, self.mel_targets, self.n_frame, self.wav]
            #self.__batch_tensors = tf.train.batch(input_tensors, batch_size=self.hp.batch_size, capacity=100,
            #                                         num_threads=10, dynamic_pad=True,
            #                                         allow_smaller_final_batch=False)

    def _get_batch_input(self):
        """返回一组训练数据输入.

        Returns:
          6元组, 包括帧数, 标签, 特征.
        """
        inputs, input_lengths, linear_targets, mel_targets, n_frame, wav = \
            tuple(self.__batch_tensors)
        return inputs, input_lengths, linear_targets, mel_targets, n_frame, wav


'''
    def read_and_decode(self, filename_queue):
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
              'data': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
          }
      )
      data = tf.decode_raw(features['data'], tf.float64)
      data = tf.cast(data, tf.float32)
      data = tf.reshape(data, [500,64,1])

      label = tf.cast(features['label'], tf.int64)

      return data, label

    def input(self, filename, batch_size):
      with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])
        image,label = read_and_decode(filename_queue)
        #print image._shape, label._shape
        images,labels = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,
                                                 capacity=10+3*batch_size)#, min_after_dequeue=10)
      return images,labels
'''





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


















