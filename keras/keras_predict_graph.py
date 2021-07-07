#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import json
import random
import numpy as np
import tensorflow as tf
import os
import cv2

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

success = 0
fails = 0


def check_file(file, postfix):
    fake_file = "."
    real = False
    if file.endswith(postfix):
        real = True
    if file.startswith(fake_file):
        real = False

    return real


def load_graph(file_path):
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="", op_dict=None,
                            producer_op_list=None)
    graph_nodes = [n for n in graph_def.node]
    return graph, graph_nodes


def work_impl(sess, input_node, output_node, files, labels, width, height, channel, task_name, debug=False):
    # for file in tqdm(files, desc=task_name):
    # print(labels)
    for file in files:
        if check_file(file, '.jpeg') is False:
            continue

        real_label = os.path.basename(os.path.dirname(file))

        # image = cv2.imread(file)
        # image = cv2.resize(image, (width, height))
        # # scale image data
        # image = image.astype("float32") / 255.0
        # # Flatten the image
        # # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # image = image.reshape((1,  width, height, channel))
        image = np.array(Image.open(file)).reshape(1, width, height, channel) / 255.0

        logits = sess.run(output_node, feed_dict={input_node: image})
        index = np.argmax(logits)
        ai_label = labels[str(index)]
        if debug:
            print('real label =', real_label, ', ai label =', ai_label, ' , file =', file)

        if real_label == ai_label:
            global success
            success += 1
        else:
            global fails
            fails += 1


def predict(model_dir='', test_dir='',
            input_tensor_name='input:0',
            output_tensor_name='mobile_net_v2/Softmax/Softmax:0',
            width=224, height=224, channel=3,
            debug=False):
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    graph, graph_nodes = load_graph(frozen_graph_file)
    # print("num nodes", len(graph_nodes))
    for node in graph_nodes:
        pass
        # print('node:', node.name)

    input_node = graph.get_tensor_by_name(input_tensor_name)
    # print("input_node:", input_node)
    output_node = graph.get_tensor_by_name(output_tensor_name)
    # print("output_node:", output_node)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        # logits = sess.run(output, feed_dict={input_node: img})

        max_per_file = 500
        files = list()
        for file in sorted(os.listdir(test_dir)):
            if os.path.isfile(os.path.join(test_dir, file)):
                files.append(os.path.join(test_dir, file))
            elif os.path.isdir(os.path.join(test_dir, file)):
                for sub_file in sorted(os.listdir(os.path.join(test_dir, file))):
                    files.append(os.path.join(test_dir, file, sub_file))

        # random.shuffle(files)
        sorted(files)
        executor = ThreadPoolExecutor(max_workers=20)

        totals = len(files)
        print('totals = ', totals)

        tasks = list()
        task_id = 0
        start = 0
        end = min(max_per_file, len(files))
        while end < len(files):
            task_name = 'work_' + '{0:03d}'.format(task_id)
            # work_impl(frozen_func, files, labels, width, height, channel, task_name, debug=False)
            args = (sess, input_node, output_node, files[start:end], labels, width, height, channel, task_name, debug)
            work_impl(*args)
            # task = executor.submit(work_impl, *args)
            # tasks.append(task)
            start += max_per_file
            end += max_per_file
            task_id += 1

        end = len(files)
        task_name = 'work_' + '{0:03d}'.format(task_id)
        args = (sess, input_node, output_node, files[start:end], labels, width, height, channel, task_name, debug)
        work_impl(*args)
        # task = executor.submit(work_impl, *args)
        # tasks.append(task)

        for future in as_completed(tasks):
            try:
                for data in future.result():
                    continue
            except Exception as exc:
                print('exception = ', exc)

        print('totals =', totals, ' , success =', success, ' , fails =', fails)
        print('percent=', format(success / totals, '.4f'))


if __name__ == '__main__':
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        model_dir = '/home/solo/code/image_classifier/out/mnist_image_classifier/20210701-1555/freeze_model/'
        test_dir = '/home/solo/code/image_classifier/resource/mnist/test/'
        predict(model_dir=model_dir,
                test_dir=test_dir,
                # input_tensor_name='x:0',
                # output_tensor_name='Identity:0',
                input_tensor_name='input:0',
                output_tensor_name='Softmax/Softmax:0',
                width=28, height=28, channel=1,
                debug=True)
    except KeyboardInterrupt:
        exit()
