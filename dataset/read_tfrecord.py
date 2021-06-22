#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


from dataset.base_tfrecord import BaseTfrecord


class ReadTfrecord(BaseTfrecord):

    def __init__(self, num_classes, train_tfrecord_list, val_tfrecord_list, test_tfrecord_list):
        super().__init__(num_classes)
        self.train_tfrecord_list = train_tfrecord_list
        self.val_tfrecord_list = val_tfrecord_list
        self.test_tfrecord_list = test_tfrecord_list
        self.num_classes = num_classes

    def get_datasets(self):
        train_dataset = self.get_dataset_from_tfrecord(self.train_tfrecord_list)
        valid_dataset = self.get_dataset_from_tfrecord(self.val_tfrecord_list)
        test_dataset = self.get_dataset_from_tfrecord(self.test_tfrecord_list)

        return train_dataset, valid_dataset, test_dataset
