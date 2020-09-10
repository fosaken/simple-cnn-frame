# -*-coding:utf-8-*-
#
#    @header utils.py
#    @author  CaoZhihui
#    @date    2020/8/31
#    @abstract:
#

import os
import sys
from importlib import import_module


def get_instance(cfg):
    assert isinstance(cfg, dict)
    assert "name" in cfg
    return eval(cfg.pop("name"))(**cfg)


def load_config_from_file(fp):
    if fp.endswith(".py"):
        dir_ = os.path.dirname(fp)
        filename = os.path.basename(fp[:fp.rfind(".")])
        sys.path.append(dir_)
        m = import_module(filename)
        sys.path.pop()
        return {name: value for name, value in m.__dict__.items() if not name.startswith("__")}
    else:
        raise NotImplementedError
