#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import optparse
import sys

import tensorflow as tf

import keras2pb
from base import time_utils
from base.log_utils import TerminalLogger
from callbacks import TrainCallback
from config.global_configs import BaseConfig, ProjectConfig, TrainBaseConfig, \
    TrainConfig, TFRecordBaseConfig, TFRecordConfig, UserConfig
from dataset.write_tfrecord import WriteTfrecord
from dataset.read_tfrecord import ReadTfrecord
from models.neural_network import NeuralNetwork


def parseargs():
    usage = 'usage: %prog [options] arg1 arg2'
    parser = optparse.OptionParser(usage=usage)

    option = optparse.OptionGroup(parser, 'region classifier trained options')

    option.add_option('-p', '--project', dest='project', type='string',
                      help='which project', default='mnist_region_classifier')
    option.add_option('-n', '--net', dest='net', type='string',
                      help='network', default=None)
    option.add_option('-t', '--time', dest='time', type='string',
                      help='time dir', default=None)
    option.add_option('-g', '--gpus', dest='gpus', type='string',
                      help='gpus', default='0,1')
    parser.add_option_group(option)

    (options, args) = parser.parse_args()

    return (options, args)


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


def train(project):
    start_time = time_utils.get_current()

    writeTfrecord = WriteTfrecord(dataset_dir=TFRecordConfig.getDefault().source_image_train_dir,
                                  dataset_test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                  tf_records_output_dir=TFRecordConfig.getDefault().tfrecord_dir,
                                  tf_records_meta_file=TFRecordConfig.getDefault().meta_file,
                                  thread=TFRecordBaseConfig.MAX_THREAD)
    out_dir = writeTfrecord.dataset_to_tfrecord()
    send_msg_to_bot(start_time, '路径 = {}'.format(out_dir))

    # step 1
    # get dataset from tfrecord
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_DATASET)
    # TFRecordConfig.getDefault().__str__()

    num_classes = TFRecordConfig.getDefault().num_classes
    readTfrecord = ReadTfrecord(num_classes=num_classes,
                                train_tfrecord_list=TFRecordConfig.getDefault().train_tfrecord_list,
                                val_tfrecord_list=TFRecordConfig.getDefault().val_tfrecord_list,
                                test_tfrecord_list=TFRecordConfig.getDefault().test_tfrecord_list)
    train_dataset, valid_dataset, test_dataset = readTfrecord.get_datasets()

    # step 2
    # init & build network(model)
    neural_network = NeuralNetwork(num_classes=num_classes,
                                   input_shape=(None, TFRecordConfig.getDefault().image_width,
                                                TFRecordConfig.getDefault().image_height,
                                                TFRecordConfig.getDefault().channels),
                                   input_tensor_name=TrainBaseConfig.INPUT_TENSOR_NAME,
                                   output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME,
                                   initial_learning_rate=TrainConfig.getDefault().initial_learning_rate,
                                   decay_steps=TrainConfig.getDefault().decay_steps,
                                   decay_rate=TrainConfig.getDefault().decay_rate,
                                   metrics=TrainBaseConfig.METRICS,
                                   network=ProjectConfig.getDefault().net)
    neural_network = neural_network.build_model()

    # step 3
    # train model
    neural_network.fit(train_dataset,
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

    # step 4
    # test model
    results = neural_network.evaluate(test_dataset)
    print("test loss, test acc:", results)

    # step 5
    # save final weights
    neural_network.save_weights(filepath=TrainConfig.getDefault().final_dir, save_format='tf')
    # save final model
    tf.keras.models.save_model(neural_network, TrainConfig.getDefault().final_dir)
    neural_network.save(TrainConfig.getDefault().final_dir, save_format='tf')
    # neural_network.save(os.path.join(TrainConfig.getDefault().final_dir, 'saved_model.h5'), save_format='h5')

    send_msg_to_bot(start_time,
                    '训练完成\ntest loss & test acc = {}\n模型路径 = {}'.format(results, TrainConfig.getDefault().model_dir))

    # # TODO
    # # step 6
    # # freeze grap
    keras2pb.freeze_session(model_dir=TrainConfig.getDefault().final_dir,
                            frozen_out_dir=TrainConfig.getDefault().model_dir,
                            frozen_graph_filename=TrainConfig.getDefault().project,
                            meta_file=TFRecordConfig.getDefault().meta_file)

    if BaseConfig.DEBUG:
        best_model_timestamp = sorted(os.listdir(TrainConfig.getDefault().train_best_export_dir))[-1]
        final_model_path = os.path.join(TrainConfig.getDefault().train_best_export_dir, best_model_timestamp)
        neural_network = tf.keras.models.load_model(final_model_path)
        print(neural_network.input)
        print(neural_network.outputs)


def main():
    time = None
    (options, args) = parseargs()
    project = options.project.strip()
    if options.time:
        time = options.time.strip()
    if options.net:
        net = options.net.strip()
    gpus = options.gpus.strip()

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        gpus = tf.config.experimental.list_physical_devices()
        if gpus:
            for gpu in gpus:
                if 'GPU' in gpu.name:
                    print(gpu)
                    tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print('exception when set gpus, error = ', e)

    # init or update configs
    ProjectConfig.getDefault().update(project=project, time=time, net=net)
    UserConfig.getDefault().update()
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_BASE)
    TrainConfig.getDefault().update()
    # TrainConfig.getDefault().__str__()

    # set log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    tf.compat.v1.get_logger().setLevel(tf.compat.v1.logging.INFO)
    # sys.stdout = TerminalLogger()
    # sys.stderr = TerminalLogger()
    terminalLogger = TerminalLogger(log_file=TrainConfig.getDefault().log_file)

    # train
    train(project)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
