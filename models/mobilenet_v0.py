#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf

from models.base_model import BaseModel


class MobileNetV0(BaseModel):
    def __init__(self, num_classes=1000, input_shape=(224, 224, 13), input_tensor_name='input',
                 output_tensor_name='Softmax'):
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=input_tensor_name)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same")
        self.separable_conv_1 = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), strides=1,
                                                                padding="same")
        self.separable_conv_2 = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(3, 3), strides=2,
                                                                padding="same")
        self.separable_conv_3 = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(3, 3), strides=1,
                                                                padding="same")
        self.separable_conv_4 = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=2,
                                                                padding="same")
        self.separable_conv_5 = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=1,
                                                                padding="same")
        self.separable_conv_6 = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(3, 3), strides=2,
                                                                padding="same")
        self.separable_conv_7 = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(3, 3), strides=2,
                                                                padding="same")
        self.separable_conv_8 = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(3, 3), strides=1,
                                                                padding="same")
        # self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax,
                                        name=output_tensor_name)

