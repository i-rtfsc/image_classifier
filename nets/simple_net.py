#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import tensorflow as tf


class SimpleNet(object):
    # self.input_layer = self.InputLayer(input_shape=input_shape, name=input_tensor_name)
    # self.flatten = self.Flatten()
    # self.relu = self.Dense(512, activation=tf.nn.relu)
    # self.fc = self.Dense(units=num_classes, activation=tf.nn.softmax, name=output_tensor_name)

    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), input_tensor_name='input',
                 output_tensor_name='Softmax'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name

    def net(self, x, is_training):
        # x = tf.placeholder(dtype=tf.float32,
        #                      shape=[None, *self.input_shape],
        #                      name=self.input_tensor_name)

        _in = tf.reshape(x, [None, *self.input_shape])

        # filters       : 输出通道的个数
        # kernel_size   : 卷积核大小, 必须是一个数字（高和宽都是此数字）或者长度为 3 的列表（分别代表高、宽）
        # strides       : 卷积步长, 默认为 (1, 1),
        # padding       : 有 valid 和 same 两种模式, 默认为 valid
        # use_bias      : 默认为True,
        _conv_2d = tf.layers.Conv2D(filters=6, kernel_size=3, strides=(1, 1), padding='same', use_bias=False)(_in)

        # 假设输入_in为[?, 28, 28, 1]
        # 然后将其传给 conv2d () 方法
        # filters 设定为 6, 即输出通道为 6
        # kernel_size 为 3, 即卷积核大小为 2 x 2
        # padding 方式设置为 same, 那么输出结果的宽高和原来一定是相同的，但是输出通道就变成了 6, 结果如下:
        # Tensor("conv2d/BiasAdd:0", shape=(?, 28, 28, 6), dtype=float32)

        # 但如果我们将 padding 方式不传入，使用默认的 valid 模式
        # 结果如下:
        # Tensor("conv2d/BiasAdd:0", shape=(?, 26, 26, 6), dtype=float32)
        # 结果就变成了 [?, 26, 26, 6], 这是因为步长默认为 1，卷积核大小为 3 x 3，所以得到的结果的高宽即为 (28 - (3 - 1)) x (28 - (3 - 1)) = 26 x 26。
        # 当然卷积核我们也可以变换大小，传入一个列表形式：
        # _conv_2d = tf.layers.Conv2D(filters=6, kernel_size=[2, 3], strides=(3, 2), padding='same', use_bias=False)(_in)
        # 这时卷积核大小变成了 2 x 3，步长变成了 3 x 2，所以结果的高宽为 ceil (28 - (2- 1)) / 3 x ceil (28 - (3- 1)) / 2 = 7 x 13，得到的结果即为 [?, 7, 13, 6]
        # 运行结果如下:
        # Tensor("conv2d/BiasAdd:0", shape=(?, 7, 13, 6), dtype=float32)

        _batch_normal = tf.layers.BatchNormalization(scale=False, center=True)(_conv_2d, training=is_training)

        _relu = tf.nn.relu(_batch_normal)

        _conv_2d_2 = tf.layers.Conv2D(filters=12, kernel_size=6, padding='same', use_bias=False, strides=2)(_relu)
        _batch_normal_2 = tf.layers.BatchNormalization(scale=False, center=True)(_conv_2d_2, training=is_training)
        _relu_2 = tf.nn.relu(_batch_normal_2)

        _conv_2d_3 = tf.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2)(_relu_2)
        _batch_normal_3 = tf.layers.BatchNormalization(scale=False, center=True)(_conv_2d_3, training=is_training)
        _relu_3 = tf.nn.relu(_batch_normal_3)

        # Tensor 进行展平操作
        _flatten = tf.layers.Flatten()(_relu_3)
        # 全连接层
        _dense = tf.layers.Dense(200, use_bias=False)(_flatten)
        _batch_normal_4 = tf.layers.BatchNormalization(scale=False, center=True)(_dense, training=is_training)
        _relu_4 = tf.nn.relu(_batch_normal_4)
        # 按照一定的概率将其暂时从网络中丢弃，可以用来防止过拟合
        _dropout = tf.layers.Dropout(0.5)(_relu_4, training=is_training)

        logits = tf.layers.Dense(self.num_classes)(_dropout)
        # predictions = tf.nn.softmax(logits)
        # classes = tf.math.argmax(predictions, axis=-1)

        return logits
