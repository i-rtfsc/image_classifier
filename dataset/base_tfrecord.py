#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf
from config.global_configs import TFRecordBaseConfig, TFRecordConfig


class BaseTfrecord(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def image_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )

    def bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature_list(self, value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # Create a dictionary with features that may be relevant.
    def create_image_example(self, image_string, label):
        feature = {
            TFRecordBaseConfig.LABEL: self.int64_feature(label),
            TFRecordBaseConfig.IMAGE: self.image_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def prepare_sample(self, features):
        image = tf.image.resize(features[TFRecordBaseConfig.IMAGE], size=(TFRecordConfig.getDefault().image_size))
        label = tf.one_hot(features[TFRecordBaseConfig.LABEL], self.num_classes)
        # label = features[TFRecordBaseConfig.LABEL]
        return image, label

    def parse_tfrecord_fn(self, example):
        feature_description = {
            TFRecordBaseConfig.IMAGE: tf.io.FixedLenFeature([], tf.string),
            TFRecordBaseConfig.LABEL: tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example[TFRecordBaseConfig.IMAGE] = tf.io.decode_jpeg(example[TFRecordBaseConfig.IMAGE],
                                                              channels=TFRecordConfig.getDefault().channels)
        return example

    def get_dataset_from_tfrecord(self, filenames):
        dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads=TFRecordBaseConfig.AUTOTUNE)
                .map(self.parse_tfrecord_fn, num_parallel_calls=TFRecordBaseConfig.AUTOTUNE)
                .map(self.prepare_sample, num_parallel_calls=TFRecordBaseConfig.AUTOTUNE)
                .shuffle(TFRecordBaseConfig.BATCH_SIZE * 10)
                .batch(TFRecordBaseConfig.BATCH_SIZE)
                .prefetch(TFRecordBaseConfig.AUTOTUNE)
        )
        return dataset
