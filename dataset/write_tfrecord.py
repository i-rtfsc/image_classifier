#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import random
import json
import tensorflow as tf

from base import file_utils
from config.global_configs import TFRecordBaseConfig, TFRecordConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.base_tfrecord import BaseTfrecord
from tqdm import tqdm


class ParallelFarm:
    def __init__(self, files, output_dir, label_dict, test_files):
        self.files = files
        self.test_files = test_files
        self.output_dir = output_dir
        self.label_dict = label_dict

        self.group_numbers = {}
        self.group_files = {}
        self.source_list = []
        self.batch_list = {}

        self.group_numbers[TFRecordBaseConfig.TRAIN] = int(
            len(files) * TFRecordBaseConfig.TRAIN_SET_RATIO)

        self.group_numbers[TFRecordBaseConfig.VAL] = int(
            len(files) * TFRecordBaseConfig.VALID_SET_RATIO)

        # self.group_numbers[TFRecordBaseConfig.TEST] = len(files) - self.group_numbers[
        #     TFRecordBaseConfig.TRAIN] - \
        #                                               self.group_numbers[TFRecordBaseConfig.VAL]
        if self.test_files:
            self.group_numbers[TFRecordBaseConfig.TEST] = len(test_files)
        else:
            self.group_numbers[TFRecordBaseConfig.TEST] = 0

        self.group_files[TFRecordBaseConfig.TRAIN] = list(self.files)[
                                                     :self.group_numbers[TFRecordBaseConfig.TRAIN]]
        self.batch_list[TFRecordBaseConfig.TRAIN] = []

        self.group_files[TFRecordBaseConfig.VAL] = list(self.files)[
                                                   self.group_numbers[TFRecordBaseConfig.TRAIN]:
                                                   self.group_numbers[
                                                       TFRecordBaseConfig.TRAIN] +
                                                   self.group_numbers[
                                                       TFRecordBaseConfig.VAL]]
        self.batch_list[TFRecordBaseConfig.VAL] = []

        # self.group_files[TFRecordBaseConfig.TEST] = list(self.files)[
        #                                             self.group_numbers[TFRecordBaseConfig.TRAIN] +
        #                                             self.group_numbers[
        #                                                 TFRecordBaseConfig.VAL]:]
        if self.test_files:
            self.group_files[TFRecordBaseConfig.TEST] = list(self.test_files)
        else:
            self.group_files[TFRecordBaseConfig.TEST] = list()
        self.batch_list[TFRecordBaseConfig.TEST] = []

    def dump_meta_json(self, extra_params=None):
        with open(TFRecordConfig.getDefault().meta_file, 'w') as f:
            content = {
                TFRecordBaseConfig.GROUP_NUMBER: self.group_numbers,
                TFRecordBaseConfig.TRAIN_TFRECORD_LIST: self.batch_list[TFRecordBaseConfig.TRAIN],
                TFRecordBaseConfig.VAL_TFRECORD_LIST: self.batch_list[TFRecordBaseConfig.VAL],
                TFRecordBaseConfig.TEST_TFRECORD_LIST: self.batch_list[TFRecordBaseConfig.TEST],
            }
            if extra_params:
                for k, v in extra_params.items():
                    content[k] = v
            json.dump(content, f, indent=4)


class WriteTfrecord(BaseTfrecord):

    def __init__(self, dataset_dir=TFRecordConfig.getDefault().source_image_train_dir,
                 dataset_test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                 tf_records_output_dir=TFRecordConfig.getDefault().tfrecord_dir,
                 tf_records_meta_file=TFRecordConfig.getDefault().meta_file,
                 thread=TFRecordBaseConfig.MAX_THREAD):
        self.dataset_dir = dataset_dir
        self.dataset_test_dir = dataset_test_dir
        self.tf_records_output_dir = tf_records_output_dir
        self.tf_records_meta_file = tf_records_meta_file
        self.thread = thread

    def work(self, files, tfrecord_name, group_numbers, group, labels_and_index):
        failed = 0
        # write the images and labels to tfrecord format file
        with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
            # for file in tqdm(files, desc=os.path.basename(tfrecord_name)):
            for file in files:
                label = labels_and_index[os.path.basename(os.path.dirname(file))]
                print("Writing to tfrecord: {}, file:{}, label:{}".format(tfrecord_name, file, label))
                image = tf.io.decode_jpeg(tf.io.read_file(file))
                try:
                    tf_example = self.create_image_example(image, label)
                    if tf_example:
                        writer.write(tf_example.SerializeToString())
                    else:
                        failed += 1
                except Exception as e:
                    print('Exception when parse, error = ', e)
                    failed += 1

        if failed > 0:
            print("%s files failed and skipped", str(failed))
            group_numbers[group] -= failed

        return str(failed)

    def dataset_to_tfrecord(self):
        if os.path.exists(self.tf_records_meta_file):
            print('tfrecord was exists!!!')
            return self.tf_records_output_dir

        task_name_tfrecords = ".tfrec"
        image_paths, image_labels, labels_and_index = file_utils.get_images_and_labels(self.dataset_dir)
        random.shuffle(image_paths)
        file_utils.create_directory(self.tf_records_output_dir)

        image_test_paths = None
        if os.path.exists(self.dataset_test_dir):
            image_test_paths, _, _ = file_utils.get_images_and_labels(self.dataset_test_dir)
            random.shuffle(image_test_paths)

        farm = ParallelFarm(image_paths, self.tf_records_output_dir, labels_and_index, image_test_paths)
        executor = ThreadPoolExecutor(max_workers=self.thread)
        tasks = []

        # loop examples and split into tasks
        for group in [TFRecordBaseConfig.TRAIN, TFRecordBaseConfig.VAL, TFRecordBaseConfig.TEST]:
            task_id = 0
            start = 0
            end = min(TFRecordBaseConfig.MAX_PER_FILE, len(farm.group_files[group]))
            while end < len(farm.group_files[group]):
                task_name = group + '_' + '{0:03d}'.format(task_id) + task_name_tfrecords
                tfrecord_file = os.path.join(self.tf_records_output_dir, task_name)
                args = (farm.group_files[group][start:end], tfrecord_file, farm.group_numbers, group, labels_and_index)
                task = executor.submit(self.work, *args)
                tasks.append(task)
                farm.batch_list[group].append(task_name)
                start += TFRecordBaseConfig.MAX_PER_FILE
                end += TFRecordBaseConfig.MAX_PER_FILE
                task_id += 1

            if len(farm.group_files[group][start:end]) > 0:
                task_name = group + '_' + '{0:03d}'.format(task_id) + task_name_tfrecords
                tfrecord_file = os.path.join(self.tf_records_output_dir, task_name)
                end = len(farm.group_files[group])
                args = (farm.group_files[group][start:end], tfrecord_file, farm.group_numbers, group, labels_and_index)
                task = executor.submit(self.work, *args)
                tasks.append(task)
                farm.batch_list[group].append(task_name)

        saved_config_list = []
        for future in as_completed(tasks):
            try:
                for result in future.result():
                    saved_config_list.append(result)
            except Exception as exc:
                print(exc)

        labels = dict()
        for k, v in labels_and_index.items():
            labels[v] = k

        farm.dump_meta_json(extra_params={TFRecordBaseConfig.TRAIN_LABELS: labels})

        return self.tf_records_output_dir
