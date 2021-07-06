#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import sys
import shutil

import tensorflow as tf

from base import time_utils
from keras import keras2pb, keras_inference_graph, keras_inference_model
from keras.callbacks import TrainCallback
from config.global_configs import ProjectConfig, TrainBaseConfig, \
    TrainConfig, TFRecordConfig, UserConfig
from net.neural_network import NeuralNetwork


def send_msg_to_bot(start_time, message):
    total_time, minutes = time_utils.elapsed_interval(start_time, time_utils.get_current())
    prepare_message = '任务耗时 = {}\n{}'.format(total_time, message)
    print(prepare_message)
    if minutes > 1800:
        bot_file = os.path.join(ProjectConfig.getDefault().root_dir, 'im/we_chat.py')
        if os.path.exists(bot_file):
            sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
            from im.we_chat import Bot
            bot = Bot(UserConfig.getDefault().bot)
            bot.set_text(prepare_message, type='text').send()
        else:
            import json
            import requests
            wx_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={}'.format(UserConfig.getDefault().bot)
            data = json.dumps({"msgtype": "text", "text": {"content": prepare_message}})
            requests.post(wx_url, data)


def train(train_dataset, valid_dataset, test_dataset, gpu):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        gpus = tf.config.experimental.list_physical_devices()
        if gpus:
            for gpu in gpus:
                if 'GPU' in gpu.name:
                    print(gpu)
                    tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print('exception when set gpus, error = ', e)

    start_time = time_utils.get_current()

    # step 1
    # init model
    neural_network = NeuralNetwork(
        network=ProjectConfig.getDefault().net,
        num_classes=TFRecordConfig.getDefault().num_classes,
        is_training=False)
    keras_network = neural_network.init_keras_network(input_shape=TFRecordConfig.getDefault().image_shape,
                                                      input_tensor_name=TrainBaseConfig.INPUT_TENSOR_NAME,
                                                      output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME,
                                                      convert=False)

    # step 2
    # train model
    keras_network.fit(train_dataset,
                      epochs=TrainConfig.getDefault().epochs,
                      validation_data=valid_dataset,
                      callbacks=TrainCallback.get_callbacks(model=neural_network,
                                                            filepath=os.path.join(
                                                                TrainConfig.getDefault().check_point_dir,
                                                                'model-epoch-{epoch:08d}-val_loss-{val_loss:.8f}'),
                                                            model_filepath=TrainConfig.getDefault().train_best_export_dir,
                                                            monitor=TrainConfig.getDefault().monitor,
                                                            min_delta=TrainConfig.getDefault().min_delta,
                                                            patience=TrainConfig.getDefault().patience,
                                                            csv_log_file=TrainConfig.getDefault().csv_log_file,
                                                            log_dir=TrainConfig.getDefault().log_dir)
                      )

    # step 3
    # test model
    results = neural_network.evaluate(test_dataset)
    print("test loss, test acc:", results)

    # step 4
    # save final weights
    neural_network.save_weights(filepath=TrainConfig.getDefault().final_dir, save_format='tf')
    # save final model
    tf.keras.models.save_model(neural_network, TrainConfig.getDefault().final_dir)
    neural_network.save(TrainConfig.getDefault().final_dir, save_format='tf')
    neural_network.save(os.path.join(TrainConfig.getDefault().final_dir, 'saved_model.h5'), save_format='h5')
    shutil.copy(TFRecordConfig.getDefault().meta_file, TrainConfig.getDefault().final_dir)

    # inference model
    keras_inference_model.inference(model_dir=TrainConfig.getDefault().final_dir,
                                    test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                    width=TFRecordConfig.getDefault().image_width,
                                    height=TFRecordConfig.getDefault().image_height,
                                    channel=TFRecordConfig.getDefault().channels,
                                    debug=True)

    send_msg_to_bot(start_time,
                    '训练完成\ntest loss & test acc = {}\n模型路径 = {}'.format(results, TrainConfig.getDefault().model_dir))

    # # TODO
    # # step 5
    # # freeze grap
    keras2pb.freeze_session(model_dir=TrainConfig.getDefault().final_dir,
                            frozen_out_dir=TrainConfig.getDefault().model_dir,
                            frozen_graph_filename=TrainConfig.getDefault().project,
                            meta_file=TFRecordConfig.getDefault().meta_file)

    # TODO
    # inference graph
    keras_inference_graph.inference(model_dir=TrainConfig.getDefault().final_dir,
                                    test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                    width=TFRecordConfig.getDefault().image_width,
                                    height=TFRecordConfig.getDefault().image_height,
                                    channel=TFRecordConfig.getDefault().channels,
                                    debug=True)

    if ProjectConfig.getDefault().debug:
        best_model_timestamp = sorted(os.listdir(TrainConfig.getDefault().train_best_export_dir))[-1]
        final_model_path = os.path.join(TrainConfig.getDefault().train_best_export_dir, best_model_timestamp)
        neural_network = tf.keras.models.load_model(final_model_path)
        print(neural_network.input)
        print(neural_network.outputs)
