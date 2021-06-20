#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import time
from datetime import datetime


def get_time_str():
    return time.strftime("%Y%m%d-%H%M", time.localtime())


def get_current():
    return datetime.now()


def elapsed_interval(start, end):
    elapsed = end - start
    min, secs = divmod(elapsed.days * 86400 + elapsed.seconds, 60)
    hour, minutes = divmod(min, 60)
    return '%.2d:%.2d:%.2d' % (hour, minutes, secs), (hour * 3600 + minutes * 60)
