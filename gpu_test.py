#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import tensorflow as tf


def test():
    tf.__version__
    print(tf.test.is_gpu_available())
    print(tf.test.gpu_device_name())


if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:
        exit()
