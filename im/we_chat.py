#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-

import types
import requests
import logging
import time
import sys
import os
import base64
import hashlib
from threading import Thread

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

DEBUG = sys.flags.debug or 'pydevd' in sys.modules
TEST = 'PYTEST_CURRENT_TEST' in os.environ


# https://work.weixin.qq.com/help?person_id=1&doc_id=13376
class Bot(Thread):

    def __init__(self, bot_key):
        super().__init__()
        self.msg_type = ''
        self.url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'.format(key=bot_key)
        self._sleep_seconds = 60
        self._check_counter = -1
        self._check_fn = None
        self._check_args = []
        self._check_kwargs = {}
        self._text = ''
        self._text_render_fn = None
        self._text_render_args = []
        self._text_render_kwargs = {}
        self._mentioned_list = []
        self._mentioned_mobile_list = []
        self._send_counter = -1
        self._image_path = ''

    def every(self, second=0, minute=0, hour=0, day=0):
        assert isinstance(second, int)
        assert isinstance(minute, int)
        assert isinstance(hour, int)
        assert isinstance(day, int)
        self._sleep_seconds = second + minute * 60 + hour * 3600 + day * 86400
        if self._sleep_seconds == 0:
            self._sleep_seconds = 60
            logging.warning("Bot.every() 解析到每 0 秒自行一次，被重设为每 60 秒")
        return self

    def check(self, fn, args=None, kwargs=None):
        assert isinstance(fn, types.FunctionType)
        assert args is None or isinstance(args, list)
        assert kwargs is None or isinstance(kwargs, dict)
        self._check_fn = fn
        self._check_args = args or []
        self._check_kwargs = kwargs or {}
        return self

    def __check(self):
        if self._check_fn:
            return self._check_fn(*self._check_args, **self._check_kwargs)
        return True

    def set_text(self, text="", type='text'):
        self.__set_content(text, type)
        return self

    def set_image_path(self, path):
        # TODO: check format of the file. Only png&jpg supported
        self.__set_content(path, type='image')
        return self

    def __set_content(self, content, type):
        assert isinstance(content, str)
        assert isinstance(type, str)
        self.msg_type = type
        if type == 'text' or type == 'markdown':
            self._text = content
        elif type == 'image':
            self._image_path = content
        return self

    def render_text(self, fn, args=None, kwargs=None, type='text'):
        assert isinstance(type, str)
        assert isinstance(fn, types.FunctionType)
        assert args is None or isinstance(args, list)
        assert kwargs is None or isinstance(kwargs, dict)
        self.msg_type = type
        self._text_render_fn = fn
        self._text_render_args = args or []
        self._text_render_kwargs = kwargs or {}
        return self

    def __get_text(self):
        if self._text_render_fn:
            return self._text_render_fn(*self._text_render_args, **self._text_render_kwargs)
        elif self._text:
            return self._text
        else:
            return ''
            # raise KeyError("请设置发送的消息")

    def set_mentioned_list(self, ls):
        assert isinstance(ls, list)
        self._mentioned_list = ls
        return self

    def set_mentioned_mobile_list(self, ls):
        assert isinstance(ls, list)
        self._mentioned_mobile_list = ls
        return self

    def set_check_counter(self, n=-1):
        assert isinstance(n, int)
        self._check_counter = n
        return self

    def set_send_counter(self, n=-1):
        assert isinstance(n, int)
        self._send_counter = n
        return self

    def send(self):
        assert self.url is not None
        if self.msg_type == '':
            raise ValueError("Empty content")
        if self.msg_type == 'text':
            return self._send_text()
        elif self.msg_type == 'markdown':
            return self._send_markdown()
        elif self.msg_type == 'image':
            return self._send_image()
        elif self.msg_type == 'news':
            raise NotImplemented
        else:
            raise TypeError('Not supported message type')

    def _send_text(self):
        req_body = {
            "msgtype": "text",
            "text": {
                "content": self.__get_text(),
                "mentioned_list": self._mentioned_list,
                "mentioned_mobile_list": self._mentioned_mobile_list
            }
        }
        if DEBUG or TEST:
            return self.url, req_body
        else:
            rsp = requests.post(self.url, json=req_body)
            if rsp.status_code != 200:
                logging.error(rsp)

    def _send_markdown(self):
        if self._mentioned_list or self._mentioned_mobile_list:
            logging.warning('Msg type Markdown does not support mentioning')
        req_body = {
            "msgtype": "markdown",
            "markdown": {
                "content": self.__get_text(),
                "mentioned_list": self._mentioned_list,
                "mentioned_mobile_list": self._mentioned_mobile_list
            }
        }
        if DEBUG or TEST:
            return self.url, req_body
        else:
            rsp = requests.post(self.url, json=req_body)
            if rsp.status_code != 200:
                logging.error(rsp)

    def _send_image(self):
        if self._mentioned_list or self._mentioned_mobile_list:
            logging.warning('Msg type Image does not support mentioning')
        # TODO: check file format
        # TODO: check file size
        image_file = open(self._image_path, 'rb')
        image_in_byte = image_file.read()
        image_file.close()
        image_b64 = base64.b64encode(image_in_byte).decode('utf-8')
        md5er = hashlib.md5()
        md5er.update(image_in_byte)
        image_md5 = md5er.digest().hex()
        req_body = {
            "msgtype": "image",
            "image": {
                "base64": image_b64,
                "md5": image_md5,
                "mentioned_list": self._mentioned_list,
                "mentioned_mobile_list": self._mentioned_mobile_list
            }
        }
        requests.post(self.url, json=req_body)

    def run(self):
        debug_msgs = []
        while self._send_counter and self._check_counter:
            self._check_counter -= 1
            if self.__check():
                if DEBUG or TEST:
                    msg = self.send()
                    debug_msgs.append(msg)
                else:
                    self.send()
                self._send_counter -= 1
            time.sleep(self._sleep_seconds)
        if DEBUG or TEST:
            return debug_msgs
