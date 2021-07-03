#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import shutil

from tensorflow.python.tools import freeze_graph

from base import env_utils, file_utils


def freeze_session(model_dir='', frozen_out_dir='',
                   frozen_graph_filename='region_classifier', output_tensor_name='Softmax',
                   gpu='0', meta_file=None):
    final_model_path = None
    for model in os.listdir(model_dir):
        if model.endswith('.pb'):
            final_model_path = model_dir

    if final_model_path is None:
        best_model_timestamp = sorted(os.listdir(model_dir))[-1]
        last_model_path = os.path.join(model_dir, best_model_timestamp)
        for model in os.listdir(last_model_path):
            if model.endswith('.pb'):
                final_model_path = last_model_path

    if final_model_path is None:
        print('error, model file was none')
        return

    file_utils.create_directory(frozen_out_dir)

    if not os.path.exists(meta_file):
        print('meta file was none, does not need copy')
    else:
        shutil.copy(meta_file, frozen_out_dir)

    output_path = os.path.join(frozen_out_dir, '{}.pb'.format(frozen_graph_filename))

    env_utils.select_gpu(gpu)

    freeze_graph.freeze_graph(input_graph=None,
                              input_saver=None,
                              input_binary=False,
                              input_checkpoint=None,
                              output_node_names=output_tensor_name,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=output_path,
                              clear_devices=False,
                              initializer_nodes=None,
                              input_saved_model_dir=final_model_path)


if __name__ == '__main__':
    pass
