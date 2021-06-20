#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import os
import datetime
import io
import pathlib
import random

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
# datasets数据集的管理, layers, optimizers优化器, sequential容器, metrics测试的度量器
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3


class MobileNetV0(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MobileNetV0, self).__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.separable_conv_1 = tf.keras.layers.SeparableConv2D(filters=64,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")

        self.separable_conv_2 = tf.keras.layers.SeparableConv2D(filters=128,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_3 = tf.keras.layers.SeparableConv2D(filters=128,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_4 = tf.keras.layers.SeparableConv2D(filters=256,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_5 = tf.keras.layers.SeparableConv2D(filters=256,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_6 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_7 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_8 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)
        self.fc = tf.keras.layers.Dense(units=7, activation=tf.keras.activations.softmax, name='Softmax')

    def call(self, inputs, training=None, mask=None):
        input = self.conv1(inputs)
        input = self.separable_conv_1(input)
        input = self.separable_conv_2(input)
        input = self.separable_conv_3(input)
        input = self.separable_conv_4(input)
        input = self.separable_conv_5(input)
        input = self.separable_conv_6(input)
        input = self.separable_conv_7(input)
        input = self.separable_conv_8(input)
        input = self.avg_pool(input)
        input = self.fc(input)

        return input


# convert a value to a type compatible tf.train.Feature
def _bytes_feature(value):
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0.))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]

    return all_image_path, all_image_label


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    # shuffle the dict
    image_paths_and_labels_dict = shuffle_dict(image_paths_and_labels_dict)
    # write the images and labels to tfrecord format file
    if not os.path.exists(os.path.dirname(tfrecord_name)):
        os.makedirs(os.path.dirname(tfrecord_name))
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, label in image_paths_and_labels_dict.items():
            print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())


def creating_tfrecords():
    dataset_to_tfrecord(dataset_dir='../resource/crop/train/', tfrecord_name='../out/test/train.tfrec')
    dataset_to_tfrecord(dataset_dir='../resource/crop/test/', tfrecord_name='../out/test/valid.tfrec')


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto.
    return tf.io.parse_single_example(example_proto, {
        'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
        'image': tf.io.FixedLenFeature([], tf.dtypes.string),
    })


def get_parsed_dataset(tfrecord_name):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    parsed_dataset = raw_dataset.map(_parse_image_function)

    return parsed_dataset


def read_tfrecords():
    batch_size = 128
    train_dataset = get_parsed_dataset(tfrecord_name='../out/test/train.tfrec')
    valid_dataset = get_parsed_dataset(tfrecord_name='../out/test/valid.tfrec')

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=batch_size)
    valid_dataset = valid_dataset.batch(batch_size=batch_size)

    return train_dataset, valid_dataset


def load_and_preprocess_image(image_raw):
    # decode
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)

    image = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image


def process_features(features):
    image_raw = features['image'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


def build_network():
    network = MobileNetV0()
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()
    return network


def train():
    creating_tfrecords()

    train_dataset, valid_dataset = read_tfrecords()

    NUM_CLASSES = 7
    EPOCHS = 2

    network = build_network()
    optimizer = optimizers.Adam(learning_rate=1e-3)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = network(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, network.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = network(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features)
            train_step(images, labels)
            # train_step(images, tf.one_hot(labels, NUM_CLASSES))
            print("train, epoch:{}, step: {},loss: {:.5f}, accuracy: {:.5f}".format(epoch, step,
                                                                                    train_loss.result().numpy(),
                                                                                    train_accuracy.result().numpy()))

        step = 0
        for features in valid_dataset:
            step += 1
            images, labels = process_features(features)
            valid_step(images, labels)
            # valid_step(images, tf.one_hot(labels, NUM_CLASSES))
            print("val, epoch:{}, step: {},loss: {:.5f}, accuracy: {:.5f}".format(epoch, step,
                                                                                  valid_loss.result().numpy(),
                                                                                  valid_accuracy.result().numpy()))


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        exit()
