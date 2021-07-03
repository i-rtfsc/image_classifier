#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf

import logging

global do_export
do_export = False

global should_early_stop
should_early_stop = False


class BetterExporter(tf.estimator.LatestExporter):
    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        global do_export
        if not do_export:
            return None
        else:
            do_export = False
            return super().export(estimator, export_path, checkpoint_path, eval_result, is_the_final_export)


# class TrainEarlyStoppingHook(tf.estimator.SessionRunHook):
#     # Hook that requests stop at a specified step.
#     # https://stackoverflow.com/questions/48815906/implement-early-stopping-in-tf-estimator-dnnregressor-using-the-available-traini
#     # https://blog.csdn.net/zaf0516/article/details/96889724
#     def __init__(self, monitor='val_loss', min_delta=0, patience=0,
#                  mode='auto'):
#         """
#         """
#         self.monitor = monitor
#         self.patience = patience
#         self.min_delta = min_delta
#         self.wait = 0
#         if mode not in ['auto', 'min', 'max']:
#             logging.warning('EarlyStopping mode %s is unknown, '
#                             'fallback to auto mode.', mode, RuntimeWarning)
#             mode = 'auto'
#
#         if mode == 'min':
#             self.monitor_op = np.less
#         elif mode == 'max':
#             self.monitor_op = np.greater
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less
#
#         if self.monitor_op == np.greater:
#             self.min_delta *= 1
#         else:
#             self.min_delta *= -1
#
#         self.best = np.Inf if self.monitor_op == np.less else -np.Inf
#
#     def begin(self):
#         # Convert names to tensors if given
#         graph = tf.get_default_graph()
#         self.monitor = graph.as_graph_element(self.monitor)
#         if isinstance(self.monitor, tf.Operation):
#             self.monitor = self.monitor.outputs[0]
#
#     def before_run(self, run_context):  # pylint: disable=unused-argument
#         # return session_run_hook.SessionRunArgs(self.monitor)
#         global should_early_stop
#         if should_early_stop:
#             print("do early stop!")
#             run_context.request_stop()
#
#     def after_run(self, run_context, run_values):
#         global do_export
#         current = run_values.results
#
#         if self.monitor_op(current - self.min_delta, self.best):
#             do_export = True
#             self.best = current
#             self.wait = 0
#         else:
#             do_export = False
#             self.wait += 1
#             if self.wait >= self.patience:
#                 global should_early_stop
#                 should_early_stop = True
#                 run_context.request_stop()


class TrainEarlyStoppingHook(tf.estimator.SessionRunHook):
    def before_run(self, run_context):
        global should_early_stop
        if should_early_stop:
            print("TrainStopHook do early stop!")
            run_context.request_stop()


class EvalEarlyStoppingHook(tf.estimator.SessionRunHook):
    def __init__(self, evaluate_steps, patience=10, total_eval_examples=0, batch_size=0):
        self._best_accuracy = 0
        self._candidate_times = 0
        self._patience = patience
        self._accuracy_tensor = None
        self._evaluate_steps = evaluate_steps
        self._total_eval_examples = total_eval_examples
        self._batch_size = batch_size
        self._step = 0
        self._stop_step = 0

    def begin(self):
        print("EarlyStoppingHook Begin")
        self._step = 0
        self._accuracy_tensor = tf.get_default_graph().get_tensor_by_name("accuracy/value:0")
        if self._total_eval_examples != 0:
            stop_step = math.ceil(self._total_eval_examples / self._batch_size)
            self._stop_step = min(stop_step, self._evaluate_steps)
        else:
            self._stop_step = self._evaluate_steps
        print("EarlyStoppingHook stop step =", self._stop_step)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({"accuracy": self._accuracy_tensor})

    def after_run(self, run_context, run_values):
        global do_export
        self._step += 1
        if self._step == self._stop_step:
            accuracy_value = run_context.session.run(self._accuracy_tensor)
            if accuracy_value > self._best_accuracy:
                print("EarlyStoppingHook reached better accuracy, assign export task.")
                do_export = True
                self._best_accuracy = accuracy_value
                self._candidate_times = 0
            else:
                do_export = False
                self._candidate_times += 1
                if self._patience > 0 and self._patience <= self._candidate_times:
                    print(
                        "EarlyStoppingHook reached best accuracy of =", self._best_accuracy, ' , Early stopping...')
                    global should_early_stop
                    should_early_stop = True
                    run_context.request_stop()
