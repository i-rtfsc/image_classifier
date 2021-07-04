#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import configparser
import json
import os

from base import time_utils
from base.switch_utils import switch, case
from base.singleton import Singleton


class BaseConfig(object):

    def update(self, *args, **kwargs):
        pass

    def __str__(self):
        print(['%s:%s' % item for item in self.__dict__.items()])


@Singleton
class ProjectConfig(BaseConfig):
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.time = time_utils.get_time_str()
        self.project = None
        self.net = 'mobilenet_v0'
        self.out = None
        self.debug = 0
        self.image_width = None
        self.image_height = None
        self.channels = None
        self.image_size = [self.image_width, self.image_height]
        self.image_shape = [self.image_width, self.image_height, self.channels]
        self.source_image_train_dir = None
        self.source_image_test_dir = None
        self.source_image_download_dir = None
        self.source_image_extract_dir = None
        print('project config class init, time =', self.time)

    # kwargs:
    # time
    # project
    # debug
    def update(self, *args, **kwargs):
        for key, value in kwargs.items():
            if 'time' == key and value is not None:
                self.time = value
            if 'project' == key and value is not None:
                self.project = value
            if 'net' == key and value is not None:
                self.net = value
            if 'debug' == key and value is not None:
                self.debug = value

        self.out = os.path.join(os.path.join(self.root_dir, 'out'), self.project, self.time)
        print('project config update config from cfg, project name =', self.project, ', time =', self.time, ', net =',
              self.net, ', debug =', self.debug)

        try:
            base_config = configparser.ConfigParser()
            file = os.path.join(self.root_dir, 'config', 'project.cfg')
            base_config.read(file)
            configs = base_config[self.project.upper()]
            self.image_width = int(configs['IMAGE_WIDTH'])
            self.image_height = int(configs['IMAGE_HEIGHT'])
            self.channels = int(configs['CHANNELS'])
            self.image_size = [self.image_width, self.image_height]
            self.image_shape = [self.image_width, self.image_height, self.channels]
            self.source_image_train_dir = os.path.join(self.root_dir, configs['SOURCE_IMAGE_TRAIN'])
            self.source_image_test_dir = os.path.join(self.root_dir, configs['SOURCE_IMAGE_TEST'])
            self.source_image_download_dir = configs['SOURCE_IMAGE_DOWNLOAD']
            self.source_image_extract_dir = configs['SOURCE_IMAGE_EXTRACT']
        except Exception as e:
            print('exception when parse, error = ', e)


@Singleton
class UserConfig(BaseConfig):
    def __init__(self):
        self.bot = None

    def update(self, *args, **kwargs):
        if self.bot is None:
            try:
                config = configparser.ConfigParser()
                config.read(os.path.join(ProjectConfig.getDefault().root_dir, 'config', 'secret.cfg'))
                self.bot = config['USER']['BOT']
                print('user config update , bot =', self.bot)
            except Exception as e:
                print('exception when parse, error ', e)


class TFRecordBaseConfig(BaseConfig):
    IMAGE = 'image'
    LABEL = 'label'

    # 每一个tfrecord的图片文件总数
    MAX_PER_FILE = 4096
    # train/valid/test数据的比例
    TRAIN_SET_RATIO = 0.9
    VALID_SET_RATIO = 0.1
    # test改成单独放在一个测试文件夹
    TEST_SET_RATIO = 1 - TRAIN_SET_RATIO - VALID_SET_RATIO
    # 生成tfrecord时的线程数
    MAX_THREAD = 20

    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    AUTOTUNE = 12
    BATCH_SIZE = 128
    BUFFER_SIZE = 30000

    META_FILE = 'meta.json'
    TRAIN_TFRECORD_LIST = 'train_tfrecord_list'
    VAL_TFRECORD_LIST = 'val_tfrecord_list'
    TEST_TFRECORD_LIST = 'test_tfrecord_list'
    TRAIN_LABELS = 'labels'
    GROUP_NUMBER = 'group_numbers'
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    UPDATE_BASE = 'update_base'
    UPDATE_DATASET = 'update_dataset'


@Singleton
class TFRecordConfig(TFRecordBaseConfig):
    def __init__(self):
        print('tfrecord init')
        # 在update函数更新
        self.source_image_train_dir = None
        self.source_image_test_dir = None
        self.image_width = None
        self.image_height = None
        self.channels = None
        self.image_size = None
        self.image_shape = None
        self.tfrecord_dir = None
        self.meta_file = None
        self.train_tfrecord_list = list()
        self.val_tfrecord_list = list()
        self.test_tfrecord_list = list()
        self.labels = None
        self.num_classes = None
        self.val_numbers = -1

    def update(self, *args, **kwargs):
        action = args[0]
        print('update tfrecord config, action =', action)
        while switch(action):
            if case(self.UPDATE_BASE):
                self.source_image_train_dir = ProjectConfig.getDefault().source_image_train_dir
                self.source_image_test_dir = ProjectConfig.getDefault().source_image_test_dir
                # image shape
                self.image_width = ProjectConfig.getDefault().image_width
                self.image_height = ProjectConfig.getDefault().image_height
                self.channels = ProjectConfig.getDefault().channels
                self.image_size = [self.image_width, self.image_height]
                self.image_shape = [self.image_width, self.image_height, self.channels]
                self.tfrecord_dir = os.path.join(ProjectConfig.getDefault().out, 'tfrecord')
                self.meta_file = os.path.join(self.tfrecord_dir, self.META_FILE)
                break

            if case(self.UPDATE_DATASET):
                with open(self.meta_file, 'r') as f:
                    meta = json.load(f)
                    self.labels = meta[self.TRAIN_LABELS]

                    for v in meta[self.TRAIN_TFRECORD_LIST]:
                        self.train_tfrecord_list.append(os.path.join(self.tfrecord_dir, v))

                    for v in meta[self.VAL_TFRECORD_LIST]:
                        self.val_tfrecord_list.append(os.path.join(self.tfrecord_dir, v))

                    for v in meta[self.TEST_TFRECORD_LIST]:
                        self.test_tfrecord_list.append(os.path.join(self.tfrecord_dir, v))

                    self.val_numbers = meta[self.GROUP_NUMBER]['val']

                self.num_classes = len(self.labels)
                break


class TrainBaseConfig(BaseConfig):
    INPUT_TENSOR_NAME = 'input'
    OUTPUT_TENSOR_NAME = 'Softmax'
    METRICS = ['accuracy']

    DROP_RATE = 0.5
    EVALUATION_STEP = 100
    EARLY_STOPPING_PATIENCE = 45
    EVAL_THROTTLE_SECS = 1000
    SAVE_CHECKPOINTS_SECS = 1000
    EVAL_START_DELAY_SECS = 500
    SHUFFLE_BUFFER_SIZE = 30000
    QUANT = 1
    QUANT_DELAY = 8000

    BEST_EXPORT = 'best'
    FINAL_EXPORT = 'final'


@Singleton
class TrainConfig(TrainBaseConfig):

    def __init__(self):
        # 在update函数更新
        self.project = None
        self.train_dir = None
        self.model_freeze_dir = None
        self.train_best_export_dir = None
        # self.check_point_dir = None
        self.final_dir = None
        self.log_dir = None
        self.log_file = None

        # 以下数据配置在train.cfg
        # 所有数据训练多少轮
        self.epochs = None
        # 初始的学习率是initial_learning_rate
        # 每decay_steps步就乘以decay_rate
        self.initial_learning_rate = None
        self.decay_steps = None
        self.decay_rate = None
        self.max_steps = 10000000
        # cks
        # loss,accuracy,val_loss,val_accuracy
        self.monitor = None
        self.min_delta = None
        self.patience = None
        # 以上数据配置在train.cfg

    def update(self, *args, **kwargs):
        print('update train config')
        self.project = ProjectConfig.getDefault().project
        self.train_dir = os.path.join(ProjectConfig.getDefault().out, 'trained')
        self.model_freeze_dir = os.path.join(ProjectConfig.getDefault().out, 'freeze_model')
        self.train_best_export_dir = os.path.join(self.train_dir, 'export', self.BEST_EXPORT)
        self.final_dir = os.path.join(self.train_dir, 'export', self.FINAL_EXPORT)

        # self.check_point_dir = os.path.join(self.train_dir, 'check_point')
        self.log_dir = os.path.join(self.train_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, 'train.log')

        try:
            base_config = configparser.ConfigParser()
            file = os.path.join(ProjectConfig.getDefault().root_dir, 'config', 'train.cfg')
            base_config.read(file)
            if ProjectConfig.getDefault().debug:
                configs = base_config['DEBUG']
            else:
                configs = base_config['TRAIN']
            self.epochs = int(configs['EPOCHS'])
            self.max_steps = int(configs['STEPS'])
            self.initial_learning_rate = float(configs['INITIAL_LEARNING_RATE'])
            self.decay_steps = int(configs['DECAY_STEPS'])
            self.decay_rate = float(configs['DECAY_RATE'])
            self.monitor = configs['MONITOR']
            self.min_delta = float(configs['MIN_DELTA'])
            self.patience = int(configs['PATIENCE'])
        except Exception as e:
            print('exception when parse, error =', e)
