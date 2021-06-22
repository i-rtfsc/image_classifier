#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-


import logging
import os
import sys


# class TerminalLogger(object):
#     def __init__(self, filename=None):
#         self.terminal = sys.stdout
#         if filename:
#             if not os.path.exists(filename):
#                 pardir = os.path.abspath(os.path.join(filename, os.pardir))
#                 if not os.path.exists(pardir):
#                     os.makedirs(pardir)
#                 file = open(filename, 'w')
#                 file.close()
#         self.log = open(filename, "a+")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
#     def close(self):
#         self.terminal.close()

class TerminalLogger(object):
    def __init__(self, level=logging.INFO, log_file=None):
        """Create a configured instance of logger."""
        fmt = '[%(asctime)s] %(levelname)s : %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt=date_fmt)

        # logger = logging.getLogger('tensorflow')
        logger = logging.getLogger()

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

        self.logger = logger

    def getLogger(self):
        return self.logger
