#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


from config.global_configs import TFRecordConfig
from dataset.base_tfrecord import BaseTfrecord


class ReadTfrecord(BaseTfrecord):

    def __init__(self):
        self.tfrecordConfig = TFRecordConfig()
        num_classes = self.tfrecordConfig.num_classes
        super().__init__(num_classes)

    def get_datasets(self):
        train_dataset = self.get_dataset_from_tfrecord(self.tfrecordConfig.train_tfrecord_list)
        valid_dataset = self.get_dataset_from_tfrecord(self.tfrecordConfig.val_tfrecord_list)
        test_dataset = self.get_dataset_from_tfrecord(self.tfrecordConfig.test_tfrecord_list)

        return train_dataset, valid_dataset, test_dataset, self.tfrecordConfig.labels
