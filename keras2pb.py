#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import shutil
import tensorflow as tf


# import tensorflow.compat.v1 as tf  # for TF 2
# tf.disable_v2_behavior()  # for TF 2


def freeze_session(model_dir='', frozen_out_dir='', frozen_graph_filename='region_classifier', meta_file=None,
                   clear_devices=True, debug=True):
    final_model_file = None
    for model in os.listdir(model_dir):
        # if model.endswith('.h5'):
        #     final_model_file = os.path.join(model_dir, model)
        if model.endswith('.pb'):
            final_model_file = model_dir

    if final_model_file is None:
        best_model_timestamp = sorted(os.listdir(model_dir))[-1]
        final_model_path = os.path.join(model_dir, best_model_timestamp)
        # model: tf.keras.Model = tf.keras.models.load_model(final_model_path)
        for model in os.listdir(final_model_path):
            # if model.endswith('.h5'):
            #     final_model_file = os.path.join(final_model_path, model)
            if model.endswith('.pb'):
                final_model_file = final_model_path

    if final_model_file is None:
        print('error, model file was none')
        return

    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.keras.backend.get_session()
    graph = session.graph
    with graph.as_default():
        model = tf.keras.models.load_model(final_model_file)
        input_names = [in_node.op.name for in_node in model.inputs]
        output_names = [out_node.op.name for out_node in model.outputs]

        if debug:
            print("input_names", input_names)
            print("output_names", output_names)

        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            if debug:
                print('--->', node.name)

        if debug:
            print("len node =", len(input_graph_def.node))

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                              output_names)

        outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
        if debug:
            print("-" * 50)
        for node in outgraph.node:
            if debug:
                print('--->>>', node.name)
        if debug:
            print("length of  node", len(outgraph.node))
        tf.io.write_graph(frozen_graph, frozen_out_dir, '{}.pb'.format(frozen_graph_filename), as_text=False)

        if not os.path.exists(meta_file):
            print('meta file was none, does not need copy')
        else:
            shutil.copy(meta_file, frozen_out_dir)


if __name__ == '__main__':
    try:
        model_path = '/home/solo/code/region_classifier/out/mnist_region_classifier/20210701-1555/trained/final/'
        frozen_out_path = '/home/solo/code/region_classifier/out/mnist_region_classifier/20210701-1555/freeze_model/'
        frozen_graph_filename = 'mnist_region_classifier'
        meta_file = '/home/solo/code/region_classifier/out/mnist_region_classifier/20210701-1555/tfrecord/meta.json'

        freeze_session(model_dir=model_path,
                       frozen_out_dir=frozen_out_path,
                       frozen_graph_filename=frozen_graph_filename,
                       meta_file=meta_file)
    except KeyboardInterrupt:
        exit()
