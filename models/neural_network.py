#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

from base.switch_utils import switch, case
from config.global_configs import BaseConfig, CNNNetWork, TFRecordConfig, TrainBaseConfig, TrainConfig
from models.mobilenet_v0 import MobileNetV0
from models.mobilenet_v1 import MobileNetV1
from models.mobilenet_v2 import MobileNetV2
from models.mobilenet_v3_large import MobileNetV3Large
from models.mobilenet_v3_small import MobileNetV3Small
from models.inception_resnet_v1 import InceptionResNetV1
from models.inception_resnet_v2 import InceptionResNetV2
from models.inception_v4 import InceptionV4
from models.simple_net import SimpleNet

from models.keras.mobile_net_v0 import KMobileNetV0

class NeuralNetwork(object):

    def __init__(self, num_classes, input_shape=(None, TFRecordConfig.getDefault().image_width,
                                                 TFRecordConfig.getDefault().image_height,
                                                 TFRecordConfig.getDefault().channels),
                 input_tensor_name=TrainBaseConfig.INPUT_TENSOR_NAME,
                 output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME,
                 initial_learning_rate=TrainConfig.getDefault().initial_learning_rate,
                 decay_steps=TrainConfig.getDefault().decay_steps,
                 decay_rate=TrainConfig.getDefault().decay_rate,
                 metrics=TrainBaseConfig.METRICS,
                 network=TrainBaseConfig.NEURAL_NETWORK):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.metrics = metrics
        self.network = network

    def build_model(self):
        """选择采用哪种卷积网络"""
        while switch(self.network):
            if case(CNNNetWork.SIMPLE_NET):
                base_model = SimpleNet(num_classes=self.num_classes,
                                       input_shape=self.input_shape[1:],
                                       input_tensor_name=self.input_tensor_name,
                                       output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.MOBILE_NET_V0):
                base_model = MobileNetV0(num_classes=self.num_classes,
                                         input_shape=self.input_shape[1:],
                                         input_tensor_name=self.input_tensor_name,
                                         output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.MOBILE_NET_V1):
                base_model = MobileNetV1(num_classes=self.num_classes,
                                         input_shape=self.input_shape[1:],
                                         input_tensor_name=self.input_tensor_name,
                                         output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.MOBILE_NET_V2):
                base_model = MobileNetV2(num_classes=self.num_classes,
                                         input_shape=self.input_shape[1:],
                                         input_tensor_name=self.input_tensor_name,
                                         output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.MOBILE_NET_V3_LARGE):
                base_model = MobileNetV3Large(num_classes=self.num_classes,
                                              input_shape=self.input_shape[1:],
                                              input_tensor_name=self.input_tensor_name,
                                              output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.MOBILE_NET_V3_SMALL):
                base_model = MobileNetV3Small(num_classes=self.num_classes,
                                              input_shape=self.input_shape[1:],
                                              input_tensor_name=self.input_tensor_name,
                                              output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.INCEPTION_RESNET_V1):
                base_model = InceptionResNetV1(num_classes=self.num_classes,
                                               input_shape=self.input_shape[1:],
                                               input_tensor_name=self.input_tensor_name,
                                               output_tensor_name=self.output_tensor_name)

                break

            if case(CNNNetWork.INCEPTION_RESNET_V2):
                base_model = InceptionResNetV2(num_classes=self.num_classes,
                                               input_shape=self.input_shape[1:],
                                               input_tensor_name=self.input_tensor_name,
                                               output_tensor_name=self.output_tensor_name)
                break

            if case(CNNNetWork.INCEPTION_V4):
                base_model = InceptionV4(num_classes=self.num_classes,
                                         input_shape=self.input_shape[1:],
                                         input_tensor_name=self.input_tensor_name,
                                         output_tensor_name=self.output_tensor_name)
                break

            ValueError('This cnn neural network is not supported at this time.')
            break

        network = tf.keras.models.Sequential()
        for layer in base_model.layers:
            network.add(layer)
            if BaseConfig.DEBUG:
                print(layer)

        network.build(input_shape=self.input_shape)

        if BaseConfig.DEBUG:
            network.summary()

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)
        network.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=self.metrics)

        return network

    def get_keras_network(self):
        input_layer = tf.keras.layers.Input(
            shape=self.input_shape[1:],
            name=self.input_tensor_name)

        # create the base pre-trained model
        while switch(self.network):
            if case(CNNNetWork.MOBILE_NET_V1):
                keras_model = KMobileNetV0(input_tensor=input_layer, weights=None,
                                                              include_top=False)
                break

            if case(CNNNetWork.MOBILE_NET_V2):
                keras_model = tf.keras.applications.MobileNetV2(input_tensor=input_layer, weights=None,
                                                                include_top=False)
                break

            if case(CNNNetWork.MOBILE_NET_V3_LARGE):
                keras_model = tf.keras.applications.MobileNetV3Large(input_tensor=input_layer, weights=None,
                                                                     include_top=False)
                break

            if case(CNNNetWork.MOBILE_NET_V3_SMALL):
                keras_model = tf.keras.applications.MobileNetV3Small(input_tensor=input_layer, weights=None,
                                                                     include_top=False)
                break

            if case(CNNNetWork.INCEPTION_RESNET_V2):
                keras_model = tf.keras.applications.InceptionResNetV2(input_tensor=input_layer, weights=None,
                                                                      include_top=False)
                break

            if case(CNNNetWork.INCEPTION_V3):
                keras_model = tf.keras.applications.InceptionV3(input_tensor=input_layer, weights=None,
                                                                include_top=False)
                break

            ValueError('This cnn neural network is not supported at this time.')
            break

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(keras_model.output)
        # let's add a fully-connected layer
        fc = tf.keras.layers.Dense(1024, activation='relu')(avg_pool)
        # and a logistic layer
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.output_tensor_name)(fc)

        # this is the model we will train
        network = tf.keras.Model(inputs=keras_model.input, outputs=predictions,
                                 name=self.network.name.lower())

        # network = tf.keras.models.Sequential(keras_model.output)

        if BaseConfig.DEBUG:
            network.summary()

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)
        network.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=self.metrics)

        return network

    def simple_model(self):
        network = tf.keras.models.Sequential()
        network.add(tf.keras.layers.InputLayer(input_shape=self.input_shape[1:], name=self.input_tensor_name))
        network.add(tf.keras.layers.Flatten())
        network.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
        network.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        if BaseConfig.DEBUG:
            network.summary()

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)
        network.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=self.metrics)
        return network
