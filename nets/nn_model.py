#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.training import moving_averages


class NNModel(object):
    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.num_classes = 1000
        self.is_training = True
        self.width_multiplier = 1
        self.scope = 'NNNet'

        for key, value in kwargs.items():
            if 'inputs' == key and value is not None:
                self.inputs = value
            if 'num_classes' == key and value is not None:
                self.num_classes = value
            if 'is_training' == key and value is not None:
                self.is_training = value
            if 'width_multiplier' == key and value is not None:
                self.width_multiplier = value
            if 'scope' == key and value is not None:
                self.scope = value

    UPDATE_OPS_COLLECTION = "_update_ops_"

    def parse_filter(self, filter_size):
        if type(filter_size) == list or type(filter_size) == tuple:
            filter_height = filter_size[0]
            filter_width = filter_size[1]
        else:
            filter_width = filter_size
            filter_height = filter_size

        return filter_height, filter_width

    def parse_stride(self, strides):
        if type(strides) == list or type(strides) == tuple:
            stride_width = strides[0]
            stride_height = strides[1]

        else:
            stride_width = strides
            stride_height = strides

        return stride_width, stride_height

    def parse_size(self, sizes):
        if type(sizes) == list or type(sizes) == tuple:
            _width = sizes[0]
            _height = sizes[1]
        else:
            _width = sizes
            _height = sizes

        return _width, _height

    # create variable
    # https://www.w3cschool.cn/tensorflow_python/tensorflow_python-st6f2ez1.html
    def create_variable(self, name, shape, initializer, dtype=tf.float32, trainable=True):
        return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)

    # batchnorm layer
    def bacthnorm(self, inputs, epsilon=1e-05, momentum=0.99, is_training=True, scope=None):
        inputs_shape = inputs.get_shape().as_list()
        params_shape = inputs_shape[-1:]
        axis = list(range(len(inputs_shape) - 1))

        with tf.variable_scope(scope):
            beta = self.create_variable("beta", params_shape,
                                        initializer=tf.zeros_initializer())
            gamma = self.create_variable("gamma", params_shape,
                                         initializer=tf.ones_initializer())
            # for inference
            moving_mean = self.create_variable("moving_mean", params_shape,
                                               initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = self.create_variable("moving_variance", params_shape,
                                                   initializer=tf.ones_initializer(), trainable=False)
        if is_training:
            mean, variance = tf.nn.moments(inputs, axes=axis)
            update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                                     mean, decay=momentum)
            update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                                         variance, decay=momentum)
            tf.compat.v1.add_to_collection(self.UPDATE_OPS_COLLECTION, update_move_mean)
            tf.compat.v1.add_to_collection(self.UPDATE_OPS_COLLECTION, update_move_variance)
        else:
            mean, variance = moving_mean, moving_variance
        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

    # depthwise conv2d layer
    def depthwise_conv2d(self, inputs, filter_size=(3, 3), output_channels=1, strides=(1, 1), scope=None):
        inputs_shape = inputs.get_shape().as_list()
        in_channels = inputs_shape[-1]
        filter_height, filter_width = self.parse_filter(filter_size)
        # shape = [filter_height * filter_width * in_channels, output_channels]
        stride_width, stride_height = self.parse_stride(strides)
        with tf.variable_scope(scope):
            filter = self.create_variable("filter", shape=[filter_height, filter_width,
                                                           in_channels, output_channels],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))

        return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, stride_width, stride_height, 1],
                                      padding="SAME", rate=[1, 1])

    # depthwise conv2d layer
    def _depthwise_conv2d(self, inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
        inputs_shape = inputs.get_shape().as_list()
        in_channels = inputs_shape[-1]
        with tf.variable_scope(scope):
            filter = self.create_variable("filter", shape=[filter_size, filter_size,
                                                           in_channels, channel_multiplier],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))

        return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                      padding="SAME", rate=[1, 1])

    # conv2d layer
    def conv2d(self, inputs, output_channels, filter_size=(1, 1), strides=(1, 1), scope=None):
        inputs_shape = inputs.get_shape().as_list()
        in_channels = inputs_shape[-1]
        filter_height, filter_width = self.parse_filter(filter_size)
        # shape = [filter_height * filter_width * in_channels, output_channels]
        stride_width, stride_height = self.parse_stride(strides)
        with tf.variable_scope(scope):
            filter = self.create_variable("filter", shape=[filter_height, filter_width,
                                                           in_channels, output_channels],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs, filter, strides=[1, stride_width, stride_height, 1],
                            padding="SAME")

    # conv2d layer
    def _conv2d(self, inputs, scope, num_filters, filter_size=1, strides=1):
        inputs_shape = inputs.get_shape().as_list()
        in_channels = inputs_shape[-1]
        with tf.variable_scope(scope):
            filter = self.create_variable("filter", shape=[filter_size, filter_size,
                                                           in_channels, num_filters],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],
                            padding="SAME")

    def depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier, filter_size=(1, 1), strides=(1, 1),
                                   is_training=True, scope=None):

        output_channels = round(num_filters * width_multiplier)

        with tf.variable_scope(scope):
            # depthwise conv2d
            dw_conv = self.depthwise_conv2d(inputs, filter_size=filter_size, strides=strides, output_channels=1,
                                            scope="depthwise_conv")
            # batchnorm
            bn = self.bacthnorm(dw_conv, epsilon=1e-05, momentum=0.99, is_training=is_training, scope='dw_bn')
            # relu
            relu = tf.nn.relu(bn)
            # pointwise conv2d (1x1)
            pw_conv = self.conv2d(relu, output_channels, filter_size=filter_size, strides=strides,
                                  scope="pointwise_conv")

            # bn
            bn = self.bacthnorm(pw_conv, epsilon=1e-05, momentum=0.99, is_training=is_training, scope='pw_bn')
            return tf.nn.relu(bn)

    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        """depthwise separable convolution 2D function"""
        num_filters = round(num_filters * width_multiplier)
        strides = 2 if downsample else 1

        with tf.variable_scope(scope):
            # depthwise conv2d
            dw_conv = self._depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            # batchnorm
            bn = self.bacthnorm(dw_conv, scope="dw_bn", is_training=self.is_training)
            # relu
            relu = tf.nn.relu(bn)
            # pointwise conv2d (1x1)
            pw_conv = self._conv2d(relu, scope="pointwise_conv", num_filters=num_filters)
            # bn
            bn = self.bacthnorm(pw_conv, scope="pw_bn", is_training=self.is_training)
            return tf.nn.relu(bn)

    # avg pool layer
    def avg_pool(self, inputs, pool_size=(3, 3), scope='avg_pool'):
        with tf.variable_scope(scope):
            _width, _height = self.parse_size(pool_size)
            return tf.nn.avg_pool2d(inputs, ksize=[1, _width, _height, 1],
                                    strides=[1, _width, _height, 1], padding="VALID")

    # fully connected layer
    def fc(self, inputs, n_out, use_bias=True, scope=None):
        inputs_shape = inputs.get_shape().as_list()
        n_in = inputs_shape[-1]
        with tf.variable_scope(scope):
            weight = self.create_variable("weight", shape=[n_in, n_out],
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            if use_bias:
                bias = self.create_variable("bias", shape=[n_out, ],
                                            initializer=tf.zeros_initializer())
                return tf.compat.v1.nn.xw_plus_b(inputs, weight, bias)
            return tf.matmul(inputs, weight)
