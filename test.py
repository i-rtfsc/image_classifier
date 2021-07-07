#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

from config.global_configs import ProjectConfig, UserConfig, TFRecordConfig, TFRecordBaseConfig, TrainConfig
from dataset.write_tfrecord import WriteTfrecord


def write():
    project = 'mnist_image_classifier'
    time = '202107033-1911'
    net = 'mobilenet_v0'
    ProjectConfig.getDefault().update(project=project, time=time, net=net)
    UserConfig.getDefault().update()
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_BASE)
    TrainConfig.getDefault().update()

    writeTfrecord = WriteTfrecord(dataset_dir=TFRecordConfig.getDefault().source_image_train_dir,
                                  dataset_test_dir=TFRecordConfig.getDefault().source_image_test_dir,
                                  tf_records_output_dir=TFRecordConfig.getDefault().tfrecord_dir,
                                  tf_records_meta_file=TFRecordConfig.getDefault().meta_file,
                                  input_shape=TFRecordConfig.getDefault().image_shape,
                                  thread=TFRecordBaseConfig.MAX_THREAD)
    writeTfrecord.dataset_to_tfrecord()


if __name__ == '__main__':
    try:
        write()
    except KeyboardInterrupt:
        exit()
