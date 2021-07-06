#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


from dataset.base_tfrecord import BaseTfrecord


class ReadTfrecord(BaseTfrecord):

    def __init__(self, num_classes, train_tfrecord_list, val_tfrecord_list, test_tfrecord_list, is_keras=0):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.is_keras = is_keras
        self.train_tfrecord_list = train_tfrecord_list
        self.val_tfrecord_list = val_tfrecord_list
        self.test_tfrecord_list = test_tfrecord_list

    def get_datasets(self):
        print('is keras =', self.is_keras)
        if self.is_keras:
            train_dataset = self.get_dataset_from_tfrecord_by_keras(self.train_tfrecord_list)
            valid_dataset = self.get_dataset_from_tfrecord_by_keras(self.val_tfrecord_list)
            test_dataset = self.get_dataset_from_tfrecord_by_keras(self.test_tfrecord_list)
        else:
            train_dataset = lambda: self.get_dataset_from_tfrecord(self.train_tfrecord_list)
            valid_dataset = lambda: self.get_dataset_from_tfrecord(self.val_tfrecord_list)
            test_dataset = lambda: self.get_dataset_from_tfrecord(self.test_tfrecord_list)
        return train_dataset, valid_dataset, test_dataset
