#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import optparse
import sys

import freeze_graph
import predict_graph
import train_image_classifier
from base import time_utils
from base.log_utils import TerminalLogger

from config.global_configs import ProjectConfig, TrainBaseConfig, TrainConfig, TFRecordBaseConfig, TFRecordConfig, \
    UserConfig
from dataset.write_tfrecord import WriteTfrecord
from dataset.read_tfrecord import ReadTfrecord


def parseargs():
    usage = 'usage: %prog [options] arg1 arg2'
    parser = optparse.OptionParser(usage=usage)

    option = optparse.OptionGroup(parser, 'image classifier trained options')

    option.add_option('-p', '--project', dest='project', type='string',
                      help='which project', default='mnist_image_classifier')
    option.add_option('-n', '--net', dest='net', type='string',
                      help='network', default='mobilenet_v0')
    option.add_option('-k', '--keras', dest='keras', type='int',
                      help='use keras', default=0)
    option.add_option('-t', '--time', dest='time', type='string',
                      help='time dir', default=None)
    option.add_option('-d', '--debug', dest='debug', type='int',
                      help='debug', default=0)
    option.add_option('-s', '--steps', dest='steps', type='int',
                      help='steps', default=None)
    option.add_option('-e', '--epochs', dest='epochs', type='int',
                      help='epochs', default=None)
    option.add_option('-g', '--gpu', dest='gpu', type='string',
                      help='gpu', default='1')
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


def main():
    time = None
    steps = None
    epochs = None
    (options, args) = parseargs()
    project = options.project.strip()
    if options.time:
        time = options.time.strip()
    if options.net:
        net = options.net.strip()
    if options.steps:
        steps = options.steps
    if options.epochs:
        epochs = options.epochs
    gpu = options.gpu.strip()
    debug = options.debug
    keras = options.keras

    print('main func, project name =', project, ', net =', net, ', keras =', keras, ', time =', time, ', steps =',
          steps, ', epochs =', epochs, ', debug =', debug)

    # step 1
    # init or update configs
    ProjectConfig.getDefault().update(project=project, keras=keras, time=time, net=net, debug=debug)
    UserConfig.getDefault().update()
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_BASE)
    TrainConfig.getDefault().update(steps=steps, epochs=epochs)
    # TrainConfig.getDefault().__str__()

    # step 2
    # init logger
    terminalLogger = TerminalLogger(log_file=TrainConfig.getDefault().log_file)
    # init start time
    start_time = time_utils.get_current()

    # step 3
    # write tfrecord if need
    writeTfrecord = WriteTfrecord(dataset_dir=TFRecordConfig.getDefault().source_image_train_dir,
                                  dataset_test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                  tf_records_output_dir=TFRecordConfig.getDefault().tfrecord_dir,
                                  tf_records_meta_file=TFRecordConfig.getDefault().meta_file,
                                  input_shape=TFRecordConfig.getDefault().image_shape,
                                  is_keras=keras,
                                  thread=TFRecordBaseConfig.MAX_THREAD)
    out_dir = writeTfrecord.dataset_to_tfrecord()
    send_msg_to_bot(start_time, '路径 = {}'.format(out_dir))

    # step 4
    # get dataset from tfrecord
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_DATASET)
    readTfrecord = ReadTfrecord(num_classes=TFRecordConfig.getDefault().num_classes,
                                is_keras=keras,
                                train_tfrecord_list=TFRecordConfig.getDefault().train_tfrecord_list,
                                val_tfrecord_list=TFRecordConfig.getDefault().val_tfrecord_list,
                                test_tfrecord_list=TFRecordConfig.getDefault().test_tfrecord_list)
    train_dataset, valid_dataset, test_dataset = readTfrecord.get_datasets()

    if keras == 1:
        from keras import keras_train_image_classifier
        # step 5
        # running train
        keras_train_image_classifier.train(train_dataset, valid_dataset, test_dataset, gpu)
        return 1


    # step 5
    # running train
    train_image_classifier.running_train(train_dataset, valid_dataset, test_dataset, gpu)

    # step 6
    # freeze & save model
    freeze_graph.freeze_session(model_dir=TrainConfig.getDefault().train_best_export_dir,
                                frozen_out_dir=TrainConfig.getDefault().model_freeze_dir,
                                frozen_graph_filename=TrainConfig.getDefault().project,
                                output_tensor_name=TrainBaseConfig.OUTPUT_TENSOR_NAME,
                                gpu=gpu,
                                meta_file=TFRecordConfig.getDefault().meta_file)

    # step 7
    # test model
    totals, success, fails, percent = predict_graph.predict(model_dir=TrainConfig.getDefault().model_freeze_dir,
                                                            test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                                            test_log=TrainConfig.getDefault().inference_file,
                                                            input_tensor_name='{}:0'.format(
                                                                    TrainBaseConfig.INPUT_TENSOR_NAME),
                                                            output_tensor_name='{}:0'.format(
                                                                    TrainBaseConfig.OUTPUT_TENSOR_NAME),
                                                            shape=TFRecordConfig.getDefault().image_shape,
                                                            gpu=gpu,
                                                            debug=True)

    send_msg_to_bot(start_time, '总数 = {} , 成功 = {} , 失败 = {} , 正确率 = {} ,'.format(totals, success, fails, percent))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
