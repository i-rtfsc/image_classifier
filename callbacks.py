#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


from tensorflow import keras
from base import time_utils
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
    def get_callbacks(model, filepath, model_filepath, monitor, min_delta, patience, csv_log_file, log_dir):
        cks = [
            # LogsCallback(model),
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
