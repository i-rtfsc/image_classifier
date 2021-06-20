#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


from dataset.write_tfrecord import WriteTfrecord
from config.global_configs import TrainConfig, TFRecordConfig


def write_tfrecord():
    writeTfrecord = WriteTfrecord()
    writeTfrecord.dataset_to_tfrecord()


if __name__ == '__main__':
    try:
        # write_tfrecord()
        print(TFRecordConfig.CHANNELS)
        TFRecordConfig.update()
        print(TFRecordConfig.CHANNELS)
    except KeyboardInterrupt:
        exit()
