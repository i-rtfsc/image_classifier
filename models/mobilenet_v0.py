#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf


class MobileNetV0(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MobileNetV0, self).__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            name='input-test')
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
        # self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)
        # self.fc = tf.keras.layers.Dense(units=7, activation=tf.keras.activations.softmax, name='Softmax')

    def call(self, inputs, training=None, mask=None):
        input = tf.keras.layers.InputLayer(input_shape=(224, 224, 3),name="input"),
        input = self.conv1(inputs)
        input = self.separable_conv_1(input)
        input = self.separable_conv_2(input)
        input = self.separable_conv_3(input)
        input = self.separable_conv_4(input)
        input = self.separable_conv_5(input)
        input = self.separable_conv_6(input)
        input = self.separable_conv_7(input)
        input = self.separable_conv_8(input)

        # input = self.avg_pool(input)
        # input = self.fc(input)

        return input

    def get_avg_pool_and_fc(self, num_classes):
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax, name='Softmax')

        return avg_pool, fc
