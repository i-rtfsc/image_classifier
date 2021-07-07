#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import functools

import tensorflow as tf
from config.global_configs import ProjectConfig, TFRecordConfig, TrainBaseConfig, TrainConfig, TFRecordBaseConfig

from base.switch_utils import switch, case


class NeuralNetwork(object):

    def __init__(self,
                 network='mobilenet_v0',
                 num_classes=1000,
                 is_training=False):

        self.network = network
        self.num_classes = num_classes
        self.is_training = is_training

    def init_network(self):
        from nets.mobilenet import mobilenet_v0
        from nets.mobilenet import mobilenet_v1
        from nets.mobilenet import mobilenet_v2
        from nets.mobilenet import mobilenet_v3

        slim = tf.contrib.slim

        networks_map = {
            'mobilenet_v0': mobilenet_v0.mobilenet_v0,
            'mobilenet_v0_075': mobilenet_v0.mobilenet_v0_075,
            'mobilenet_v0_050': mobilenet_v0.mobilenet_v0_050,
            'mobilenet_v0_025': mobilenet_v0.mobilenet_v0_025,

            'mobilenet_v1': mobilenet_v1.mobilenet_v1,
            'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
            'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
            'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,

            'mobilenet_v2': mobilenet_v2.mobilenet,
            'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
            'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,

            'mobilenet_v3_small': mobilenet_v3.small,
            'mobilenet_v3_large': mobilenet_v3.large,
            'mobilenet_v3_small_minimalistic': mobilenet_v3.small_minimalistic,
            'mobilenet_v3_large_minimalistic': mobilenet_v3.large_minimalistic,
        }

        arg_scopes_map = {
            'mobilenet_v0': mobilenet_v0.mobilenet_v0_arg_scope,
            'mobilenet_v0_075': mobilenet_v0.mobilenet_v0_arg_scope,
            'mobilenet_v0_050': mobilenet_v0.mobilenet_v0_arg_scope,
            'mobilenet_v0_025': mobilenet_v0.mobilenet_v0_arg_scope,

            'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
            'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,

            'mobilenet_v2': mobilenet_v2.training_scope,
            'mobilenet_v2_035': mobilenet_v2.training_scope,
            'mobilenet_v2_140': mobilenet_v2.training_scope,

            'mobilenet_v3_small': mobilenet_v3.training_scope,
            'mobilenet_v3_large': mobilenet_v3.training_scope,
            'mobilenet_v3_small_minimalistic': mobilenet_v3.training_scope,
            'mobilenet_v3_large_minimalistic': mobilenet_v3.training_scope,

        }

        if self.network not in networks_map:
            raise ValueError(self.network, ' neural network is not supported at this time.')
        func = networks_map[self.network]

        @functools.wraps(func)
        def network_fn(images, **kwargs):
            arg_scope = arg_scopes_map[self.network](is_training=self.is_training)
            with slim.arg_scope(arg_scope):
                return func(images, self.num_classes, is_training=self.is_training, **kwargs)

        if hasattr(func, 'default_image_size'):
            network_fn.default_image_size = func.default_image_size

        return network_fn

    def init_keras_network_without_build(self, input_shape=TFRecordConfig.getDefault().image_shape,
                                         input_tensor_name=TrainBaseConfig.INPUT_TENSOR_NAME,
                                         output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME
                                         ):
        from keras.nets.simple_net import SimpleNet
        from keras.nets.mobilenet.mobilenet_v0 import MobileNetV0
        from keras.nets.mobilenet.mobilenet_v1 import MobileNetV1

        while switch(self.network):
            if case('simple_net'):
                base_model = SimpleNet(num_classes=self.num_classes,
                                       input_shape=input_shape,
                                       input_tensor_name=input_tensor_name,
                                       output_tensor_name=output_tensor_name)
                break
            if case('mobilenet_v0'):
                base_model = MobileNetV0(num_classes=self.num_classes,
                                         input_shape=input_shape,
                                         input_tensor_name=input_tensor_name,
                                         output_tensor_name=output_tensor_name)
                break

            if case('mobilenet_v1'):
                base_model = MobileNetV1(num_classes=self.num_classes,
                                         input_shape=input_shape,
                                         input_tensor_name=input_tensor_name,
                                         output_tensor_name=output_tensor_name)
                break

            ValueError('This cnn neural network is not supported at this time.')
            break

        network = tf.keras.models.Sequential()
        for layer in base_model.layers:
            network.add(layer)

        return network

    def init_keras_network(self, input_shape=TFRecordConfig.getDefault().image_shape,
                           input_tensor_name=TrainBaseConfig.INPUT_TENSOR_NAME,
                           output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME,
                           convert=True):
        network = self.init_keras_network_without_build(input_shape=input_shape,
                                                        input_tensor_name=input_tensor_name,
                                                        output_tensor_name=output_tensor_name)
        network.build(input_shape=input_shape)
        network.summary()

        if convert:
            loss = 'categorical_crossentropy'
        else:
            loss = tf.losses.CategoricalCrossentropy(from_logits=True)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=TrainConfig.getDefault().initial_learning_rate,
            decay_steps=TrainConfig.getDefault().decay_steps,
            decay_rate=TrainConfig.getDefault().decay_rate)
        network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss=loss,
                        metrics=TrainBaseConfig.METRICS)

        return network

    @staticmethod
    def build_network(features, labels, mode, params):
        network_name = ProjectConfig.getDefault().net
        num_classes = params.num_classes
        IMAGE = params.image
        LABEL = params.label
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Define model structure
        neural_network = NeuralNetwork(
            network=network_name,
            num_classes=num_classes,
            is_training=is_training
        )
        network = neural_network.init_network()

        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_keep_prob = 1 - params.drop_rate
        else:
            dropout_keep_prob = 1

        logits, endpoints = network(features[IMAGE], dropout_keep_prob=dropout_keep_prob)

        # Define the loss functions
        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            global_step = tf.train.get_or_create_global_step()
            label_one_hot = tf.one_hot(labels[LABEL], num_classes)
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=label_one_hot, logits=logits))
            # tf.summary.scalar('softmax_cross_entropy', loss)

        predictions = tf.argmax(tf.nn.softmax(logits), axis=-1, name="final_output")

        # now return these EstimatorSpec
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

            # Define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            train_op = optimizer.minimize(loss, global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if params.quant and mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        # Additional metrics for monitoring
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=labels[LABEL], predictions=predictions)
            tf.summary.scalar('accuracy', accuracy)
            eval_metric_ops = {
                "accuracy": accuracy
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        # Generate predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions_dict = {
                "predictions": predictions
            }
            export_outputs = {
                "predict_output": tf.estimator.export.PredictOutput(predictions_dict)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    @staticmethod
    def build_keras_network(features, labels, mode, params):
        network_name = ProjectConfig.getDefault().net
        num_classes = params.num_classes
        input_shape = params.shape
        input_tensor_name = params.input_tensor_name
        output_tensor_name = params.output_tensor_name
        IMAGE = params.image
        LABEL = params.label

        neural_network = NeuralNetwork(
            network=network_name,
            num_classes=num_classes,
        )
        network = neural_network.init_keras_network_without_build(
            input_shape=input_shape,
            input_tensor_name=input_tensor_name,
            output_tensor_name=output_tensor_name)

        # logits = tf.layers.dense(inputs=network.predict(features[TFRecordBaseConfig.IMAGE]),
        #                          units=num_classes)
        # https://stackoverflow.com/questions/48295788/using-a-keras-model-inside-a-tf-estimator
        feature_map = network(features)
        logits = tf.keras.layers.Dense(units=num_classes)(feature_map)

        # Define the loss functions
        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            global_step = tf.train.get_or_create_global_step()
            label_one_hot = tf.one_hot(labels[LABEL], num_classes)
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=label_one_hot, logits=logits))
            tf.summary.scalar('softmax_cross_entropy', loss)

        predictions = tf.argmax(tf.nn.softmax(logits), axis=-1, name="final_output")

        # now return these EstimatorSpec
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

            # Define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            train_op = optimizer.minimize(loss, global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if params.quant and mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        # Additional metrics for monitoring
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=labels[LABEL],
                                           predictions=predictions)
            tf.summary.scalar('accuracy', accuracy)
            eval_metric_ops = {
                "accuracy": accuracy
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        # Generate predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions_dict = {
                "predictions": predictions
            }
            export_outputs = {
                "predict_output": tf.estimator.export.PredictOutput(predictions_dict)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
