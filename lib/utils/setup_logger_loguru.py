#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import inspect
import os
import sys
import logging
from loguru import logger
import time
from collections import Counter


class InterceptHandler(logging.Handler):
    # https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_intercept():
    logging.basicConfig(handlers=[InterceptHandler()], level=0)


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a", log_level="DEBUG"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """

    def formatter(record):
        level_name = record["level"].name
        level_name_map = {
            "INFO": "INF",
            "WARNING": "WRN",
            "ERROR": "ERR",
            "DEBUG": "DBG",
        }
        level_abbr = level_name_map.get(level_name, level_name)

        caller_name = record["name"]
        if caller_name.startswith("detectron2."):
            caller_abbr = caller_name.replace("detectron2.", "d2.")
        else:
            caller_abbr = caller_name
        loguru_format = (
            "<green>{time:YYYYMMDD_HHmmss}</green>|"
            "<lvl>%s</lvl>|"
            "<cyan>%s</cyan>@<cyan>{line}</cyan>: <lvl>{message}</lvl>"
            "\n{exception}"
        ) % (level_abbr, caller_abbr)
        return loguru_format

    logger.remove()  # Remove the pre-configured handler
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=formatter,
            level=log_level,
            enqueue=True,
        )
        logger.add(save_file)

        setup_intercept()

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")
