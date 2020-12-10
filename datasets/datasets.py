# -*-coding:utf-8-*-
#
#    @header datasets.py
#    @author  CaoZhihui
#    @date    2020/9/7
#    @abstract:
#
import os

from torch.utils.data import Dataset

from .augmentations import *


class Pipeline(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class FileListDataset(Dataset):
    def __init__(self, file_list_fp, prefix="", pipelines=()):
        super().__init__()
        self._prefix = prefix
        self._parse_file_list(file_list_fp)
        self.pipeline = Pipeline(*[self._get_instance(cfg) for cfg in pipelines])

    def __getitem__(self, index):
        splits = self.lines[index].split()
        fp = os.path.join(self._prefix, splits[0])
        # TODO >>>
        # label = splits[1:]
        # label = list(map(int, label))
        label = int(splits[1])
        # <<<
        img = cv2.imread(fp)
        img = self.pipeline(img)
        # return img, np.array(label), fp
        return img, label, fp

    def _parse_file_list(self, fp):
        with open(fp, "r") as f:
            self.lines = f.read().strip().split("\n")

    @staticmethod
    def _get_instance(cfg):
        assert isinstance(cfg, dict)
        assert "name" in cfg
        return eval(cfg.pop("name"))(**cfg)

    def __len__(self):
        return len(self.lines)
