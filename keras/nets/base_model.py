#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import tensorflow as tf


class BaseModel(object):
    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), input_tensor_name='input',
                 output_tensor_name='Softmax'):
        pass

    @property
    def InputLayer(self):
        return tf.keras.layers.InputLayer

    @property
    def Flatten(self):
        return tf.keras.layers.Flatten

    @property
    def Dense(self):
        return tf.keras.layers.Dense

    @property
    def Conv2D(self):
        return tf.keras.layers.Conv2D

    @property
    def SeparableConv2D(self):
        return tf.keras.layers.SeparableConv2D

    @property
    def AveragePooling2D(self):
        return tf.keras.layers.AveragePooling2D

    @property
    def GlobalAveragePooling2D(self):
        return tf.keras.layers.GlobalAveragePooling2D

    @property
    def layers(self):
        return [v for k, v in self.__dict__.items()]
