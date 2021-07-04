#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf

from easydict import EasyDict as edict

from base import env_utils
from config.global_configs import TrainBaseConfig, \
    TrainConfig, TFRecordBaseConfig, TFRecordConfig, ProjectConfig
from hook_and_exporter import BetterExporter, EvalEarlyStoppingHook, TrainEarlyStoppingHook
from models.neural_network import NeuralNetwork


def init_training_params():
    training_params = edict(
        {
            "drop_rate": TrainBaseConfig.DROP_RATE,
            "learning_rate": TrainConfig.getDefault().initial_learning_rate,
            "decay_steps": TrainConfig.getDefault().decay_steps,
            "decay_rate": TrainConfig.getDefault().decay_rate,
            "num_classes": TFRecordConfig.getDefault().num_classes,
            "batch_size": TFRecordBaseConfig.BATCH_SIZE,
            "evaluation_step": TrainBaseConfig.EVALUATION_STEP,
            "max_steps": TrainConfig.getDefault().max_steps,
            "early_stopping_patience": TrainBaseConfig.EARLY_STOPPING_PATIENCE,
            "eval_throttle_secs": TrainBaseConfig.EVAL_THROTTLE_SECS,
            "save_checkpoints_secs": TrainBaseConfig.SAVE_CHECKPOINTS_SECS,
            "eval_start_delay_secs": TrainBaseConfig.EVAL_START_DELAY_SECS,
            "shuffle_buffer_size": TrainBaseConfig.SHUFFLE_BUFFER_SIZE,
            "quant": TrainBaseConfig.QUANT,
            "quant_delay": TrainBaseConfig.QUANT_DELAY
        }
    )
    return training_params


def serving_input_receiver_fn():
    # This is used to define inputs to serve the model.
    reciever_tensors = {
        # The size of input image is flexible.
        TFRecordBaseConfig.IMAGE: tf.placeholder(dtype=tf.float32,
                                                 shape=[None, *TFRecordConfig.getDefault().image_shape],
                                                 name=TrainBaseConfig.INPUT_TENSOR_NAME)}

    # # Convert give inputs to adjust to the model.
    # features = {
    #     # Resize given images.
    #     TFRecordBaseConfig.IMAGE: tf.reshape(reciever_tensors[INPUT_FEATURE], [None, *TFRecordConfig.getDefault().image_shape])
    # }

    # return: ServingInputReciever
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors, features=reciever_tensors.copy())


def running_train(train_dataset, valid_dataset, test_dataset, gpu='0'):
    # init training params
    training_params = init_training_params()

    # init gpu device
    env_utils.select_gpu(gpu)

    # limit to num_cpu_core CPU usage
    # tf.compat.v1.ConfigProto
    session_config = tf.ConfigProto(device_count={"CPU": 8},
                                    log_device_placement=True,
                                    inter_op_parallelism_threads=2,
                                    intra_op_parallelism_threads=5)
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig(session_config=session_config,
                                              save_checkpoints_secs=training_params.save_checkpoints_secs)

    # build estimator

    if ProjectConfig.getDefault().keras:
        # # TODO
        # # 这种训练方式遇到一个问题，暂时不会保存PB文件，待解决
        # neural_network = NeuralNetwork(
        #     network=ProjectConfig.getDefault().net,
        #     num_classes=TFRecordConfig.getDefault().num_classes,
        # )
        # keras_network = neural_network.init_keras_network()
        #
        # # 如果用这种方式训练，返回的dataset是image, classifier
        # # readTfrecord.get_keras_datasets()
        # estimator_network = tf.keras.estimator.model_to_estimator(
        #     keras_model=keras_network,
        #     model_dir=TrainConfig.getDefault().train_dir,
        #     custom_objects=training_params,
        #     config=estimator_config,
        # )

        # 如果用这种方式训练，返回的dataset是{xxx, image}, {yyy, classifier}
        # 所以如果用这种方式训练，则ReadTfrecord读dataset要使用get_dataset_from_tfrecord
        # 这种训练方式output_tensor_name不是设置的Softmax
        # TODO
        # 1. 待研究维护不是 Softmax
        # 2. 通过tf.get_default_graph().get_operations()找出output是什么
        estimator_network = tf.estimator.Estimator(
            model_fn=NeuralNetwork.build_keras_network,
            model_dir=TrainConfig.getDefault().train_dir,
            config=estimator_config,
            params=training_params,
        )
    else:
        # 如果用这种方式训练，返回的dataset是{xxx, image}, {yyy, classifier}
        # estimator都是这种数据
        estimator_network = tf.estimator.Estimator(
            model_fn=NeuralNetwork.build_network,
            model_dir=TrainConfig.getDefault().train_dir,
            config=estimator_config,
            params=training_params,
        )

    # train_early_stopping_hook = KerasTrainEarlyStoppingHook(monitor=TrainConfig.getDefault().monitor,
    #                                                    min_delta=TrainConfig.getDefault().min_delta,
    #                                                    patience=TrainConfig.getDefault().patience)
    train_early_stopping_hook = TrainEarlyStoppingHook()

    eval_train_early_stopping_hook = EvalEarlyStoppingHook(training_params.evaluation_step,
                                                           patience=training_params.early_stopping_patience,
                                                           total_eval_examples=TFRecordConfig.getDefault().val_numbers,
                                                           batch_size=training_params.batch_size)

    # # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/estimator/BestExporter
    # better_exporter = tf.estimator.BestExporter(name=TrainBaseConfig.BEST_EXPORT,
    #                                             serving_input_receiver_fn=serving_input_receiver_fn, exports_to_keep=5)
    better_exporter = BetterExporter(TrainBaseConfig.BEST_EXPORT,
                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                     exports_to_keep=5)

    final_exporter = tf.estimator.FinalExporter(TrainBaseConfig.FINAL_EXPORT,
                                                serving_input_receiver_fn=serving_input_receiver_fn)

    # train and evaluate
    train_spec = tf.estimator.TrainSpec(train_dataset,
                                        max_steps=training_params.max_steps,
                                        hooks=[train_early_stopping_hook],
                                        )
    eval_spec = tf.estimator.EvalSpec(valid_dataset, steps=training_params.evaluation_step,
                                      start_delay_secs=training_params.eval_start_delay_secs,
                                      throttle_secs=training_params.eval_throttle_secs,
                                      exporters=[final_exporter, better_exporter],
                                      hooks=[eval_train_early_stopping_hook],
                                      )
    tf.estimator.train_and_evaluate(estimator_network, train_spec, eval_spec)
