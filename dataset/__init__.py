#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import

from .kitti import *

__factory = {
    'kitti': Kitti,
}

def get_names():
    return __factory.keys()

def init_dataset(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**kwargs)