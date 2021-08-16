'''
Author: your name
Date: 2021-08-16 10:53:39
LastEditTime: 2021-08-16 11:18:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/local_utils/init_logger.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 下午9:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : init_logger.py
# @IDE: PyCharm
"""
Log relative utils
"""
import os.path as ops
import time
import loguru
from model.utils import cli_helper

CFG = cli_helper.parse_args()


def get_logger(log_file_name_prefix):
    """

    :param log_file_name_prefix: log文件名前缀
    :return:
    """
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                               time.localtime(time.time()))
    log_file_name = '{:s}_{:s}.log'.format(log_file_name_prefix, start_time)
    log_file_path = ops.join(CFG.save, log_file_name)

    logger = loguru.logger
    log_level = 'INFO'
    logger.add(log_file_path,
               level=log_level,
               format="{time} {level} {message}",
               retention="10 days",
               rotation="1 week")

    return logger

# if __name__ == '__main__':
#     logger = get_logger('test')
#     logger.info('Hello, world!')