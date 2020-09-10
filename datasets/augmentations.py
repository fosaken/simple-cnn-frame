# -*-coding:utf-8-*-
#
#    @header augmentations.py
#    @author  CaoZhihui
#    @date    2020/8/31
#    @abstract:
#

import random

import cv2
import torch
import numpy as np


class Resize(object):
    def __init__(self, scale):
        self.scale = None
        if isinstance(scale, float):
            assert 0 < scale <= 1.
            self.dsize = (0, 0)
            self.scale = scale
            pass
        elif isinstance(scale, tuple):
            self.dsize = scale

    def __call__(self, img):
        if not self.scale:
            img = cv2.resize(img, dsize=self.dsize, interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, dsize=self.dsize, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        return img


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        crop_w, crop_h = self.size
        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        return img[top:bottom, left:right]


class Normalize(object):
    def __init__(self, mean=None, std=None, permute=False):
        if not mean:
            mean = [123.675, 116.28, 103.53]
        if not std:
            std = [58.395, 57.12, 57.375]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.mean = np.float64(mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(std.reshape(1, -1))

        self.permute = permute

    def __call__(self, img):
        img_ = img.astype(np.float32)
        cv2.subtract(img_, self.mean, img_)
        cv2.multiply(img_, self.std_inv, img_)
        if self.permute:
            img_ = img_.transpose([2, 0, 1])
        return img_


class Flip(object):
    def __init__(self, order=1, ratio=0.5):
        try:
            assert order in (0, 1)
        except AssertionError:
            raise ValueError(f"Expect order in (0, 1), got {order} instead.")
        self.order = order
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < self.ratio:
            img = cv2.flip(img, flipCode=self.order)
        return img


class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img)
