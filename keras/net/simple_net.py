#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf

from keras.net.base_model import BaseModel


class SimpleNet(BaseModel):
    def __init__(self, num_classes=1000, input_shape=(224, 224, 13), input_tensor_name='input',
                 output_tensor_name='Softmax'):
        self.input_layer = self.InputLayer(input_shape=input_shape, name=input_tensor_name)
        self.flatten = self.Flatten()
        self.relu = self.Dense(512, activation=tf.nn.relu)
        self.fc = self.Dense(units=num_classes, activation=tf.nn.softmax, name=output_tensor_name)
