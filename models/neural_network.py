#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import functools

import tensorflow as tf
from config.global_configs import TFRecordBaseConfig, ProjectConfig, TFRecordConfig

from models import mobilenet_v0
from models import mobilenet_v1

slim = tf.contrib.slim


class NeuralNetwork(object):

    def __init__(self,
                 network='mobilenet_v0',
                 num_classes=1000,
                 is_training=False):

        self.network = network
        self.num_classes = num_classes
        self.is_training = is_training

        self.networks_map = {
            'mobilenet_v0': mobilenet_v0.mobilenet_v0,
            'mobilenet_v0_075': mobilenet_v0.mobilenet_v0_075,
            'mobilenet_v0_050': mobilenet_v0.mobilenet_v0_050,
            'mobilenet_v0_025': mobilenet_v0.mobilenet_v0_025,

            'mobilenet_v1': mobilenet_v1.mobilenet_v1,
            'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
            'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
            'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,

            # 'mobilenet_v2': mobilenet_v2.mobilenet,
            # 'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
            # 'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,

            # 'mobilenet_v3_small': mobilenet_v3.small,
            # 'mobilenet_v3_large': mobilenet_v3.large,
        }

        self.arg_scopes_map = {
            'mobilenet_v0': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v0_075': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v0_050': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v0_025': mobilenet_v1.mobilenet_v1_arg_scope,

            'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,

            # 'mobilenet_v2': mobilenet_v2.training_scope,
            # 'mobilenet_v2_035': mobilenet_v2.training_scope,
            # 'mobilenet_v2_140': mobilenet_v2.training_scope,

            # 'mobilenet_v3_small': mobilenet_v3.training_scope,
            # 'mobilenet_v3_large': mobilenet_v3.training_scope,
        }

    def init_network(self):

        if self.network not in self.networks_map:
            raise ValueError(self.network, ' neural network is not supported at this time.')
        func = self.networks_map[self.network]

        @functools.wraps(func)
        def network_fn(images, **kwargs):
            arg_scope = self.arg_scopes_map[self.network](is_training=self.is_training)
            with slim.arg_scope(arg_scope):
                return func(images, self.num_classes, is_training=self.is_training, **kwargs)

        if hasattr(func, 'default_image_size'):
            network_fn.default_image_size = func.default_image_size

        return network_fn

    # https://tensorflow.juejin.im/get_started/custom_estimators.html
    @staticmethod
    def build_network(features, labels, mode, params):
        images = features[TFRecordBaseConfig.IMAGE]

        neural_network = NeuralNetwork(
            network=ProjectConfig.getDefault().net,
            num_classes=TFRecordConfig.getDefault().num_classes,
        )
        network = neural_network.init_network()

        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_keep_prob = 1 - params.drop_rate
        else:
            dropout_keep_prob = 1

        logits, endpoints = network(images, dropout_keep_prob=dropout_keep_prob)

        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            global_step = tf.train.get_or_create_global_step()
            label_one_hot = tf.one_hot(labels[TFRecordBaseConfig.LABEL], params.num_classes)
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=label_one_hot, logits=logits))
            tf.summary.scalar('cross_entropy', loss)

        predictions = tf.argmax(tf.nn.softmax(logits), axis=-1, name="final_output")

        if mode == tf.estimator.ModeKeys.TRAIN:
            decay_learning_rate = tf.train.exponential_decay(
                learning_rate=params.learning_rate,
                global_step=global_step,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate
            )
            tf.summary.scalar('learning_rate', decay_learning_rate)
            if params.quant:
                g = tf.get_default_graph()
                tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=params.quant_delay)

            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            train_op = optimizer.minimize(loss, global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if params.quant and mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=labels[TFRecordBaseConfig.LABEL], predictions=predictions)
            tf.summary.scalar('accuracy', accuracy)
            eval_metric_ops = {
                "accuracy": accuracy
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions_dict = {
                "predictions": predictions
            }
            export_outputs = {
                "predict_output": tf.estimator.export.PredictOutput(predictions_dict)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
