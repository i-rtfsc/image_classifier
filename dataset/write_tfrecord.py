#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os
import random
import json
import tensorflow as tf

from base import file_utils
from config.global_configs import TFRecordConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.base_tfrecord import BaseTfrecord
from tqdm import tqdm


class ParallelFarm:
    def __init__(self, files, output_dir, label_dict):
        self.files = files
        self.output_dir = output_dir
        self.label_dict = label_dict

        self.group_numbers = {}
        self.group_files = {}
        self.source_list = []
        self.batch_list = {}

        self.group_numbers[TFRecordConfig.TRAIN] = int(len(files) * TFRecordConfig.TRAIN_SET_RATIO)

        self.group_numbers[TFRecordConfig.VAL] = int(len(files) * TFRecordConfig.VALID_SET_RATIO)

        self.group_numbers[TFRecordConfig.TEST] = len(files) - self.group_numbers[TFRecordConfig.TRAIN] - \
                                                  self.group_numbers[TFRecordConfig.VAL]

        self.group_files[TFRecordConfig.TRAIN] = list(self.files)[:self.group_numbers[TFRecordConfig.TRAIN]]
        self.batch_list[TFRecordConfig.TRAIN] = []

        self.group_files[TFRecordConfig.VAL] = list(self.files)[
                                               self.group_numbers[TFRecordConfig.TRAIN]:self.group_numbers[
                                                                                            TFRecordConfig.TRAIN] +
                                                                                        self.group_numbers[
                                                                                            TFRecordConfig.VAL]]
        self.batch_list[TFRecordConfig.VAL] = []

        self.group_files[TFRecordConfig.TEST] = list(self.files)[
                                                self.group_numbers[TFRecordConfig.TRAIN] + self.group_numbers[
                                                    TFRecordConfig.VAL]:]

        self.batch_list[TFRecordConfig.TEST] = []

    def dump_meta_json(self, extra_params=None):
        with open(TFRecordConfig.META_DIR, 'w') as f:
            content = {
                TFRecordConfig.GROUP_NUMBER: self.group_numbers,
                TFRecordConfig.TRAIN_TFRECORD_LIST: self.batch_list[TFRecordConfig.TRAIN],
                TFRecordConfig.VAL_TFRECORD_LIST: self.batch_list[TFRecordConfig.VAL],
                TFRecordConfig.TEST_TFRECORD_LIST: self.batch_list[TFRecordConfig.TEST],
            }
            if extra_params:
                for k, v in extra_params.items():
                    content[k] = v
            json.dump(content, f, indent=4)


class WriteTfrecord(BaseTfrecord):

    def __init__(self, dataset_dir=TFRecordConfig.SOURCE_IMAGE_TRAIN,
                 tf_records_output_dir=TFRecordConfig.TFRECORD_DIR,
                 thread=TFRecordConfig.MAX_THREAD):
        self.dataset_dir = dataset_dir
        self.tf_records_output_dir = tf_records_output_dir
        self.thread = thread

    def work(self, files, tfrecord_name, group_numbers, group, labels_and_index):
        failed = 0
        # write the images and labels to tfrecord format file
        with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
            for file in tqdm(files, desc=os.path.basename(tfrecord_name)):
            # for file in files:
                label = labels_and_index[os.path.basename(os.path.dirname(file))]
                # print("Writing to tfrecord: {}, file:{}, label:{}".format(tfrecord_name, file, label))
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
        task_name_tfrecords = ".tfrec"
        image_paths, image_labels, labels_and_index = file_utils.get_images_and_labels(self.dataset_dir)
        random.shuffle(image_paths)
        file_utils.create_directory(self.tf_records_output_dir)

        farm = ParallelFarm(image_paths, self.tf_records_output_dir, labels_and_index)
        executor = ThreadPoolExecutor(max_workers=self.thread)
        tasks = []

        # loop examples and split into tasks
        for group in [TFRecordConfig.TRAIN, TFRecordConfig.VAL, TFRecordConfig.TEST]:
            task_id = 0
            start = 0
            end = min(TFRecordConfig.MAX_PER_FILE, len(farm.group_files[group]))
            while end < len(farm.group_files[group]):
                task_name = group + '_' + '{0:03d}'.format(task_id) + task_name_tfrecords
                tfrecord_file = os.path.join(self.tf_records_output_dir, task_name)
                args = (farm.group_files[group][start:end], tfrecord_file, farm.group_numbers, group, labels_and_index)
                task = executor.submit(self.work, *args)
                tasks.append(task)
                farm.batch_list[group].append(task_name)
                start += TFRecordConfig.MAX_PER_FILE
                end += TFRecordConfig.MAX_PER_FILE
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

        farm.dump_meta_json(extra_params={TFRecordConfig.TRAIN_LABELS: labels})

        return self.tf_records_output_dir
