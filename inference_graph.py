#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import json
import random
import shutil
import tensorflow as tf
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

from base import env_utils, file_utils
from base.log_utils import TerminalLogger
from config.global_configs import ProjectConfig, TrainBaseConfig, TrainConfig, TFRecordBaseConfig, TFRecordConfig, \
    UserConfig

totals = 0
success = 0
fails = 0
percent = 0
result = dict()
result['success'] = list()
result['error'] = list()


def image_process(file, shape, rect=None):
    if file_utils.check_file(file, '.jpeg') is False:
        return None
    else:
        image = Image.open(file)
        (width, height) = image.size

        if rect is not None:
            left = rect['left']
            up = rect['up']
            right = rect['right']
            bottom = rect['bottom']
        else:
            left = 0
            up = 0
            right = width
            bottom = height

        image = image.crop((left, up, right, bottom))
        if shape[2] == 3:
            image = image.convert('RGB')

        image = image.resize(shape[:-1])  # 放到目标大小
        img_data = np.reshape(image, shape)
        img_data = img_data - 127.5
        img_data = img_data / 127.5

        return img_data


def work_impl(sess, predict_softmax_tensor, input_tensor, task_name, files, labels, shape, rect=None, debug=False):
    # for file in tqdm(files, desc=task_name):
    for file in files:
        real_label = file_utils.get_last_directory(file)
        image = image_process(file, shape, rect)
        logits = sess.run(predict_softmax_tensor, feed_dict={input_tensor: [image]})

        index = np.argmax(logits)
        ai_label = labels[str(index)]
        if debug:
            print('real label =', real_label, ', ai label =', ai_label, ' , prob =', logits[0][index])

        if real_label == ai_label:
            global success
            success += 1
            result['success'].append([real_label, ai_label, file, logits])
        else:
            global fails
            fails += 1
            result['error'].append([real_label, ai_label, file, logits])

    return ''


def inference(model_dir='',
              test_dir='',
              test_log='',
              input_tensor_name='input:0',
              output_tensor_name='Softmax:0',
              shape=(240, 108, 3),
              rect=None,
              gpu='2',
              debug=False):
    print(shape)
    frozen_graph_file = None
    label_file = None
    for model in os.listdir(model_dir):
        if model.endswith('.pb'):
            frozen_graph_file = os.path.join(model_dir, model)
        if model.endswith('.json'):
            label_file = os.path.join(model_dir, model)

    if not os.path.exists(test_dir):
        return None
    if not os.path.exists(frozen_graph_file):
        return None
    if not os.path.exists(label_file):
        return None

    labels = []
    with open(label_file, 'r') as file:
        _data = json.load(file)
        labels = _data['labels']

    env_utils.select_gpu(gpu)

    model_graph = tf.Graph()
    with model_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_file, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            _ = tf.import_graph_def(graph_def, name='')

    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=session_config, graph=model_graph) as sess:
        input_tensor = model_graph.get_tensor_by_name(input_tensor_name)
        predict_softmax_tensor = model_graph.get_tensor_by_name(output_tensor_name)

        max_per_file = 500
        files = list()
        for file in sorted(os.listdir(test_dir)):
            if os.path.isfile(os.path.join(test_dir, file)):
                files.append(os.path.join(test_dir, file))
            elif os.path.isdir(os.path.join(test_dir, file)):
                for sub_file in sorted(os.listdir(os.path.join(test_dir, file))):
                    files.append(os.path.join(test_dir, file, sub_file))

        random.shuffle(files)
        executor = ThreadPoolExecutor(max_workers=20)
        totals = len(files)

        tasks = list()
        task_id = 0
        start = 0
        end = min(max_per_file, len(files))
        while end < len(files):
            task_name = 'work_' + '{0:03d}'.format(task_id)
            args = (sess, predict_softmax_tensor, input_tensor, task_name, files[start:end], labels, shape, rect, debug)
            # work_impl(*args)
            task = executor.submit(work_impl, *args)
            tasks.append(task)
            start += max_per_file
            end += max_per_file
            task_id += 1

        end = len(files)
        task_name = 'work_' + '{0:03d}'.format(task_id)
        args = (sess, predict_softmax_tensor, input_tensor, task_name, files[start:end], labels, shape, rect, debug)
        # work_impl(*args)
        task = executor.submit(work_impl, *args)
        tasks.append(task)

        for future in as_completed(tasks):
            try:
                for data in future.result():
                    continue
            except Exception as exc:
                print('exception = ', exc)

    percent = format(success / totals, '.4f')
    print('totals =', totals, ' , success =', success, ' , fails =', fails, ' , percent =', percent)

    return totals, success, fails, percent


if __name__ == '__main__':
    project = 'rank_region_classifier'
    time = '20210705-1718'

    ProjectConfig.getDefault().update(project=project, time=time)
    UserConfig.getDefault().update()
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_BASE)
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_DATASET)
    TrainConfig.getDefault().update()

    inference(model_dir=TrainConfig.getDefault().model_freeze_dir,
              test_dir=TFRecordConfig.getDefault().source_image_test_dir,
              test_log=TrainConfig.getDefault().inference_file,
              input_tensor_name='{}:0'.format(TrainBaseConfig.INPUT_TENSOR_NAME),
              output_tensor_name='{}:0'.format(TrainBaseConfig.OUTPUT_TENSOR_NAME),
              shape=TFRecordConfig.getDefault().image_shape,
              gpu=2,
              debug=True)
