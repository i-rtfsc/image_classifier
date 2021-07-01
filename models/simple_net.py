#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf

from models.base_model import BaseModel


class SimpleNet(BaseModel):
    def __init__(self, num_classes=1000, input_shape=(224, 224, 13), input_tensor_name='input',
                 output_tensor_name='Softmax'):
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=input_tensor_name)
        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax, name=output_tensor_name)
