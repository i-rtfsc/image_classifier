#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf

from nets.nn_model import NNModel


class SmartNet(NNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, output_tensor_name):
        # with tf.variable_scope(self.scope):
        # conv1
        net = self.conv2d(self.inputs, output_channels=round(32 * self.width_multiplier), filter_size=(4, 2),
                          strides=(4, 2), scope="conv_1")
        print(net)

        net = tf.nn.relu(
            self.bacthnorm(net, epsilon=1e-05, momentum=0.99, is_training=self.is_training, scope='conv_1/bn'))
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=64,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(1, 1),
                                              strides=(1, 1),
                                              is_training=self.is_training, scope='ds_conv_2')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=128,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(3, 3),
                                              strides=(1, 1),
                                              is_training=self.is_training,
                                              scope='ds_conv_3')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=128,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(3, 3),
                                              strides=(1, 1),
                                              is_training=self.is_training,
                                              scope='ds_conv_4')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=256,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(3, 3),
                                              strides=(1, 1),
                                              is_training=self.is_training,
                                              scope='ds_conv_5')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=256,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(1, 1),
                                              strides=(1, 1),
                                              is_training=self.is_training,
                                              scope='ds_conv_6')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=512,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(3, 3),
                                              strides=(3, 3),
                                              is_training=self.is_training,
                                              scope='ds_conv_7')
        print(net)

        net = self.depthwise_separable_conv2d(inputs=net,
                                              num_filters=512,
                                              width_multiplier=self.width_multiplier,
                                              filter_size=(3, 3),
                                              strides=(1, 1),
                                              is_training=self.is_training,
                                              scope='ds_conv_8')
        print(net)

        net = self.avg_pool(net, pool_size=(7, 6), scope='avg_pool')
        print(net)

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        print(net)

        logits = self.fc(inputs=net, n_out=self.num_classes, scope="fc")
        print(logits)
        logits = tf.nn.softmax(logits, name=output_tensor_name)
        print(logits)

        predictions = tf.argmax(logits, axis=-1)
        print(predictions)

        return logits, predictions

    def build_mobile_network(self, output_tensor_name):
        # construct model
        with tf.variable_scope(self.scope):
            # conv1
            net = self._conv2d(inputs, "conv_1", round(32 * self.width_multiplier), filter_size=3,
                               strides=2)
            print(net)

            net = tf.nn.relu(self.bacthnorm(net, scope="conv_1/bn", is_training=self.is_training))
            print(net)

            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier, "ds_conv_2")
            print(net)

            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_3", downsample=True)
            print(net)

            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_4")
            print(net)

            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_5", downsample=True)
            print(net)

            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_6")
            print(net)

            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_7", downsample=True)
            print(net)

            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_8", downsample=True)
            print(net)

            # net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_9")
            # print(net)

            # net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_10")
            # print(net)
            #
            # net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_11")
            # print(net)
            #
            # net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_12")
            # print(net)
            #
            # net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier, "ds_conv_13", downsample=True)
            # print(net)
            #
            # net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier, "ds_conv_14")
            # print(net)

            shape = net.shape
            print(shape)

            net = self.avg_pool(net, (shape[1], shape[2]), "avg_pool")
            print(net)

            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            print(net)

            logits = self.fc(inputs=net, n_out=self.num_classes, scope="fc")
            print(logits)

            logits = tf.nn.softmax(logits, name=output_tensor_name)
            print(logits)

            predictions = tf.argmax(logits, axis=-1)
            print(predictions)

            return logits, predictions


if __name__ == '__main__':
    inputs = tf.random_normal(shape=[1, 224, 112, 1])

    smart_net = SmartNet(inputs=inputs,
                         num_classes=7,
                         is_training=True,
                         width_multiplier=1,
                         scope='SmartNet')
    logits, predictions = smart_net.build_mobile_network('Softmax')
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        pred = sess.run(predictions)
        print(logits)
        print(predictions)
