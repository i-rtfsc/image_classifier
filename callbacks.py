#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import os

from tensorflow import keras
from base import time_utils
from config.global_configs import TrainConfig
from ts_callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard


class LogsCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_train_batch_end(self, batch, logs=None):
        print(" -> on train batch end, for batch {} <- {}".format(batch, time_utils.get_current()))

    def on_test_batch_end(self, batch, logs=None):
        print(" -> on test batch end, for batch {}, <- {}".format(batch, time_utils.get_current()))

    def on_epoch_end(self, epoch, logs=None):
        print(" -> on epoch end, epoch : {}, <- {}".format(epoch, time_utils.get_current()))


class TrainCallback:

    @staticmethod
    def get_callbacks(model):
        cks = [
            # LogsCallback(model),

            ModelCheckpoint(
                filepath=os.path.join(TrainConfig.CHECK_POINT_DIR,
                                      'model-epoch-{epoch:02d}-val_loss-{val_loss:.3f}'),
                model_filepath=os.path.join(TrainConfig.TRAIN_BEST_EXPORT_DIR),
                monitor=TrainConfig.MONITOR,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                period=1),

            EarlyStopping(monitor=TrainConfig.MONITOR,
                          min_delta=TrainConfig.MIN_DELTA,
                          patience=TrainConfig.PATIENCE,
                          mode='auto'),

            CSVLogger(os.path.join(TrainConfig.LOG_DIR, 'training.log')),

            # tensorboard --logdir=trained/logs/ --port=9893 --bind_all
            TensorBoard(log_dir=TrainConfig.LOG_DIR)
        ]
        return cks
