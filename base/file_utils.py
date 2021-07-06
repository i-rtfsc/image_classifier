#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import pathlib


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_last_directory(file):
    return os.path.basename(os.path.dirname(file))


def check_file(file, postfix):
    fake_file = "."
    real = False
    if file.endswith(postfix):
        real = True
    if file.startswith(fake_file):
        real = False

    return real


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))

    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]

    return all_image_path, all_image_label, label_to_index
