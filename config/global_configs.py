#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import configparser
import json
import os
from enum import Enum, auto

from base import time_utils
from base.singleton import Singleton


class CNNNetWork(Enum):
    MOBILE_NET_V0 = auto()
    MOBILE_NET_V1 = auto()
    MOBILE_NET_V2 = auto()
    MOBILE_NET_V3_LARGE = auto()
    MOBILE_NET_V3_SMALL = auto()
    INCEPTION_RESNET_V1 = auto()
    INCEPTION_RESNET_V2 = auto()
    INCEPTION_V4 = auto()


class BaseConfig(object):
    DEBUG = True
    BOT = ''
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    SOURCE_IMAGE_TRAIN = os.path.join(ROOT_DIR, 'resource/rank/train/')
    SOURCE_IMAGE_TEST = os.path.join(ROOT_DIR, os.pardir, 'resource/rank/test/')

    @staticmethod
    def update():
        try:
            config = configparser.ConfigParser()
            config.read(os.path.join(BaseConfig.ROOT_DIR, 'config', 'secret.cfg'))
            BaseConfig.BOT = config['USER']['BOT']
            print('base config update , bot = ', BaseConfig.BOT)
        except Exception as e:
            print('exception when parse, error = ', e)


class MnistConfig(BaseConfig):
    IS_MNIST = True
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    CHANNELS = 1
    IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]
    MNIST_IMAGE_DOWNLOAD = os.path.join(BaseConfig.ROOT_DIR, 'resource/mnist/download/')
    MNIST_IMAGE_EXTRACT = os.path.join(BaseConfig.ROOT_DIR, 'resource/mnist/extract/')
    MNIST_IMAGE_TRAIN = os.path.join(BaseConfig.ROOT_DIR, 'resource/mnist/train/')
    MNIST_IMAGE_TEST = os.path.join(BaseConfig.ROOT_DIR, os.pardir, 'resource/mnist/test/')


@Singleton
class OutConfig(BaseConfig):
    def __init__(self):
        self.time = time_utils.get_time_str()
        # self.time = '20210618-1455'
        self.model_name = 'region_classifier'
        self.out = os.path.join(os.path.join(self.ROOT_DIR, 'out'), self.model_name, self.time)


class TFRecordConfig(BaseConfig):
    IMAGE = 'image'
    LABEL = 'label'

    # 生成tfrecord时的线程数
    MAX_THREAD = 20

    SOURCE_IMAGE_TRAIN = BaseConfig.SOURCE_IMAGE_TRAIN
    # image shape
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    CHANNELS = 3
    IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]

    # 每一个tfrecord的图片文件总数
    MAX_PER_FILE = 4096
    # train/valid/test数据的比例
    TRAIN_SET_RATIO = 0.8
    VALID_SET_RATIO = 0.1
    TEST_SET_RATIO = 1 - TRAIN_SET_RATIO - VALID_SET_RATIO

    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    AUTOTUNE = 12

    BATCH_SIZE = 128

    TFRECORD_DIR = os.path.join(OutConfig.instance().out, 'tfrecord')
    # TFRECORD_DIR = 'out/region_classifier/20210620-1637/tfrecord'
    META_FILE = 'meta.json'
    META_DIR = os.path.join(TFRECORD_DIR, META_FILE)

    TRAIN_TFRECORD_LIST = 'train_tfrecord_list'
    VAL_TFRECORD_LIST = 'val_tfrecord_list'
    TEST_TFRECORD_LIST = 'test_tfrecord_list'
    TRAIN_LABELS = 'labels'

    GROUP_NUMBER = 'group_numbers'
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    @staticmethod
    def update():
        BaseConfig.update()
        print('update tfrecord config')
        if MnistConfig.IS_MNIST:
            TFRecordConfig.IMAGE_WIDTH = MnistConfig.IMAGE_WIDTH
            TFRecordConfig.IMAGE_HEIGHT = MnistConfig.IMAGE_HEIGHT
            TFRecordConfig.CHANNELS = MnistConfig.CHANNELS
            TFRecordConfig.IMAGE_SIZE = MnistConfig.IMAGE_SIZE
            TFRecordConfig.SOURCE_IMAGE_TRAIN = MnistConfig.MNIST_IMAGE_TRAIN

    def __init__(self):
        print('tfrecord init')
        self.train_tfrecord_list = list()
        self.val_tfrecord_list = list()
        self.test_tfrecord_list = list()
        self.labels = None

        with open(TFRecordConfig.META_DIR, 'r') as f:
            meta = json.load(f)
            self.labels = meta[TFRecordConfig.TRAIN_LABELS]

            for v in meta[TFRecordConfig.TRAIN_TFRECORD_LIST]:
                self.train_tfrecord_list.append(os.path.join(TFRecordConfig.TFRECORD_DIR, v))

            for v in meta[TFRecordConfig.VAL_TFRECORD_LIST]:
                self.val_tfrecord_list.append(os.path.join(TFRecordConfig.TFRECORD_DIR, v))

            for v in meta[TFRecordConfig.TEST_TFRECORD_LIST]:
                self.test_tfrecord_list.append(os.path.join(TFRecordConfig.TFRECORD_DIR, v))

        self.num_classes = len(self.labels)


class TrainConfig(BaseConfig):
    MODEL_NAME = OutConfig.instance().model_name
    NEURAL_NETWORK = CNNNetWork.MOBILE_NET_V2

    TRAIN_DIR = os.path.join(OutConfig.instance().out, 'trained')
    MODEL_DIR = os.path.join(OutConfig.instance().out, 'freeze_model')
    TRAIN_BEST_EXPORT_DIR = os.path.join(TRAIN_DIR, 'best_export')
    CHECK_POINT_DIR = os.path.join(TRAIN_DIR, 'check_point')
    FINAL_DIR = os.path.join(TRAIN_DIR, 'final')
    LOG_DIR = os.path.join(TRAIN_DIR, 'logs')
    LOG_FILE = os.path.join(LOG_DIR, 'train.log')

    GPU = '0'
    # 所有数据训练多少轮
    EPOCHS = 1000000

    # 初始的学习率是1e-5
    # 每5000步就乘以0.85
    INITIAL_LEARNING_RATE = 1e-5
    DECAY_STEPS = 5000
    DECAY_RATE = 0.85

    # cks
    # loss,accuracy,val_loss,val_accuracy
    MONITOR = 'val_loss'
    MIN_DELTA = 0.0001
    PATIENCE = 100

    @staticmethod
    def update():
        BaseConfig.update()
        print('update train config')
        # 如果是debug模式，则更改参数
        if BaseConfig.DEBUG:
            TrainConfig.EPOCHS = 2
            # 把学习率改大，学习速度更快
            TrainConfig.INITIAL_LEARNING_RATE = 1e-3

    def __init__(self):
        pass
        # with open(TFRecordConfig.META_DIR, 'r') as f:
        #     meta = json.load(f)
        #     self.labels = meta[TFRecordConfig.TRAIN_LABELS]
        # self.num_classes = len(self.labels)
