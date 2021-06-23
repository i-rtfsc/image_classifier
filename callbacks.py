#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from base import time_utils
from ts_callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard


class LogsCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(time_utils.get_current(), ' -> current decayed lr = {:0.12f}'.format(current_decayed_lr))


class TrainCallback:

    @staticmethod
    def get_callbacks(model, filepath, model_filepath, monitor, min_delta, patience, csv_log_file, log_dir):
        cks = [
            LogsCallback(model),

            ModelCheckpoint(
                filepath=filepath,
                model_filepath=model_filepath,
                monitor=monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                period=1),

            EarlyStopping(monitor=monitor,
                          min_delta=min_delta,
                          patience=patience,
                          mode='auto'),

            CSVLogger(csv_log_file),

            # tensorboard --logdir=trained/logs/ --port=9893 --bind_all
            TensorBoard(log_dir=log_dir)
        ]
        return cks
