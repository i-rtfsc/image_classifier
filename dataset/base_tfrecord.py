#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-
import random

import tensorflow as tf

from config.global_configs import TFRecordBaseConfig, TFRecordConfig


class BaseTfrecord(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def image_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )

    def int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create_image_example(self, image_string, label):
        feature = {
            TFRecordBaseConfig.IMAGE: self.image_feature(image_string),
            TFRecordBaseConfig.LABEL: self.int64_feature(label),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_tfrecord_fn(self, example):
        keys_to_features = {
            TFRecordBaseConfig.IMAGE: tf.FixedLenFeature((), tf.string, default_value=""),
            TFRecordBaseConfig.LABEL: tf.FixedLenFeature((), tf.int64, default_value=0),
        }
        parsed = tf.parse_single_example(example, keys_to_features)

        img = tf.decode_raw(parsed[TFRecordBaseConfig.IMAGE], tf.uint8)
        img = tf.reshape(img, [TFRecordConfig.getDefault().image_width, TFRecordConfig.getDefault().image_height,
                               TFRecordConfig.getDefault().channels])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.multiply(tf.subtract(img, 0.5), 2)
        label = tf.cast(parsed[TFRecordBaseConfig.LABEL], tf.int32)

        features = {TFRecordBaseConfig.IMAGE: img}
        labels = {TFRecordBaseConfig.LABEL: label}
        return features, labels

    def get_dataset_from_tfrecord(self, filenames, shuffle=True):
        random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_tfrecord_fn, num_parallel_calls=TFRecordBaseConfig.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=TFRecordBaseConfig.BUFFER_SIZE)
            dataset = dataset.repeat()

        dataset = dataset.batch(TFRecordBaseConfig.BATCH_SIZE).prefetch(TFRecordBaseConfig.BATCH_SIZE)

        return dataset.make_one_shot_iterator().get_next()
