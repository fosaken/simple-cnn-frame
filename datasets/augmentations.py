# -*-coding:utf-8-*-
#
#    @header augmentations.py
#    @author  CaoZhihui
#    @date    2020/8/31
#    @abstract:
#

import math
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


class RandomResizeCrop(object):
    """
    """

    def __init__(self, size, scale=(0.5625, 1), ratio=(3. / 4, 4. / 3)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max). "
                             f"But received {scale}")

        self.scale = scale
        self.ratio = ratio

    def get_params(self, img):
        height, width = img.shape[:2]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            target_width = width
            target_height = int(round(target_width / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            target_height = height
            target_width = int(round(target_height * max(self.ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, img):
        xmin, ymin, target_height, target_width = self.get_params(img)
        img = img[ymin:ymin + target_height, xmin:xmin + target_width]
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
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
    def __init__(self, mean=None, std=None, to_rgb=True, permute=False):
        if not mean:
            mean = [123.675, 116.28, 103.53]
        if not std:
            std = [58.395, 57.12, 57.375]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.mean = np.float64(mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(std.reshape(1, -1))

        self.to_rgb = to_rgb
        self.permute = permute

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
