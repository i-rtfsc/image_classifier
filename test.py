#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import os
import tensorflow as tf

if __name__ == '__main__':
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        gpus = tf.config.experimental.list_physical_devices()
        if gpus:
            for gpu in gpus:
                if 'GPU' in gpu.name:
                    print(gpu)
                    tf.config.experimental.set_memory_growth(gpu, True)
    except KeyboardInterrupt:
        exit()
