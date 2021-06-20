#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import logging
import os


def get_logger(level=logging.DEBUG, log_file=None):
    """Create a configured instance of logger."""
    fmt = '[%(asctime)s] %(levelname)s : %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    logger = logging.getLogger("tensorflow")

    if log_file:
        if not os.path.exists(log_file):
            pardir = os.path.abspath(os.path.join(log_file, os.pardir))
            if not os.path.exists(pardir):
                os.makedirs(pardir)
            file = open(log_file, 'w')
            file.close()
        fh = logging.FileHandler(filename=log_file, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(level)
    logger.info("logger get or created.")
    return logger
