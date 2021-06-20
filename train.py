#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import sys

import tensorflow as tf

from base import time_utils, log_utils
from callbacks import TrainCallback
from config.global_configs import BaseConfig, OutConfig, TrainConfig, TFRecordConfig, MnistConfig
from dataset.write_tfrecord import WriteTfrecord
from dataset.read_tfrecord import ReadTfrecord
from models.neural_network import NeuralNetwork


def send_msg_to_bot(start_time, message):
    total_time, minutes = time_utils.elapsed_interval(start_time, time_utils.get_current())
    prepare_message = '任务耗时 = {}\n{}'.format(total_time, message)
    print(prepare_message)
    if minutes > 1800:
        bot_file = os.path.join(BaseConfig.ROOT_DIR, 'im/we_chat.py')
        if os.path.exists(bot_file):
            sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
            from im.we_chat import Bot
            bot = Bot(BaseConfig.BOT)
            bot.set_text(prepare_message, type='text').send()
        else:
            import json
            import requests
            wx_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={}'.format(BaseConfig.BOT)
            data = json.dumps({"msgtype": "text", "text": {"content": prepare_message}})
            requests.post(wx_url, data)


def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    OutConfig.instance()
    TFRecordConfig.update()
    TrainConfig.update()

    start_time = time_utils.get_current()
    writeTfrecord = WriteTfrecord(dataset_dir=TFRecordConfig.SOURCE_IMAGE_TRAIN,
                                  tf_records_output_dir=TFRecordConfig.TFRECORD_DIR,
                                  thread=TFRecordConfig.MAX_THREAD)
    out_dir = writeTfrecord.dataset_to_tfrecord()
    send_msg_to_bot(start_time, '路径 = {}'.format(out_dir))
    logger = log_utils.get_logger(log_file=TrainConfig.LOG_FILE)
    logger.info(out_dir)

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # gpus = tf.config.experimental.list_physical_devices()
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # step 1
    # get dataset from tfrecord
    readTfrecord = ReadTfrecord()
    train_dataset, valid_dataset, test_dataset, labels = readTfrecord.get_datasets()

    # step 2
    # init & build network(model)
    neural_network = NeuralNetwork(num_classes=len(labels), network=TrainConfig.NEURAL_NETWORK)
    neural_network = neural_network.build_model()

    # step 3
    # train model
    neural_network.fit(train_dataset,
                       epochs=TrainConfig.EPOCHS,
                       validation_data=valid_dataset,
                       callbacks=TrainCallback.get_callbacks(neural_network)
                       )

    # step 4
    # test model
    results = neural_network.evaluate(test_dataset)
    print("test loss, test acc:", results)

    # step 5
    # save final weights
    neural_network.save_weights(filepath=TrainConfig.FINAL_DIR, save_format='tf')
    # save final model
    tf.keras.models.save_model(neural_network, TrainConfig.FINAL_DIR)

    # TODO
    # step 6
    # freeze grap
    # freeze_graph.freeze_pb_from_model(model_path=TrainConfig.TRAIN_BEST_EXPORT_DIR,
    #                                   frozen_out_path=TrainConfig.MODEL_DIR,
    #                                   frozen_graph_filename=TrainConfig.MODEL_NAME)
    send_msg_to_bot(start_time, '训练完成\ntest loss & test acc = {}\n模型路径 = {}'.format(results, TrainConfig.MODEL_DIR))

    if BaseConfig.DEBUG:
        best_model_timestamp = sorted(os.listdir(TrainConfig.TRAIN_BEST_EXPORT_DIR))[-1]
        final_model_path = os.path.join(TrainConfig.TRAIN_BEST_EXPORT_DIR, best_model_timestamp)
        neural_network = tf.keras.models.load_model(final_model_path)
        print(neural_network.input)
        print(neural_network.outputs)


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        exit()
