# -*-coding:utf-8-*-
#
#    @header runner.py
#    @author  CaoZhihui
#    @date    2020/8/23
#    @abstract:
#

import os
import logging

import torch
from torch.utils.data import DataLoader

from .datasets import FileListDataset
from .optimizers import *
from .losses import *
from .models import *


class Optimizer(object):
    def __init__(self, optimizer, criterion):
        self._optimizer = optimizer
        self._criterion = criterion
        self._loss = None

    def update(self, predict, target, backward=True):
        self._loss = self._criterion(predict, target)
        if backward:
            self._optimizer.zero_grad()
            self._loss.backward()
            self._optimizer.step()
        return self._loss.item()


class Trainer(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.save_dir = kwargs.pop("output_dir", "tmp")

        self.eval_interval = kwargs.pop("eval_interval", 1)
        self.log_interval = kwargs.pop("log_interval", 10)

        self.gpu = kwargs.pop("use_gpu", True)
        self.gpu_ids = kwargs.pop("gpu_list", [0])

        self.start_epoch = kwargs.pop("start_epoch", 1)
        self.epochs = kwargs.pop("epochs", 100)
        self.epoch = 1

        self._build_logger()
        self._build_optimizer(kwargs.pop("optm_cfg"), kwargs.pop("loss_cfg"))
        self._build_train_dataset(kwargs.pop("train_dataset_cfg"))
        self._build_train_dataloader(kwargs.pop("train_dataloader_cfg"))
        self._build_val_dataset(kwargs.pop("val_dataset_cfg", {}))
        self._build_val_dataloader(kwargs.pop("val_dataloader_cfg", {}))

    def train(self):
        for idx, self.epoch in enumerate(range(self.start_epoch, self.epochs+1)):
            self.train_epoch()
            if (idx + 1) % self.eval_interval == 0:
                self.val()

    def train_epoch(self):
        self.model.train()
        for idx, (imgs, labels, _) in enumerate(self._train_dataloader):
            if self.gpu:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
            output = self.model(imgs)
            loss = self._optimizer.update(output, labels)
            if (idx + 1) % self.log_interval == 0:
                self.logger.info(f"Epoch:[{self.epoch}/{self.epochs}]\t [{idx+1}/{len(self._train_dataloader)}]\t Loss:{loss:.6f}")

    def val(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (imgs, labels, _) in enumerate(self._val_dataloader):
                outputs = self.model(imgs)
                val_loss += self._optimizer.update(outputs, labels, backward=False)
        self.logger.info(f"Epoch:[{self.epoch}/{self.epochs}] Val Loss:{val_loss:.6f}")

    def _build_optimizer(self, optm_cfg, loss_cfg):
        optm_cfg["params"] = self.model.parameters()
        self._optimizer = Optimizer(optimizer=self._get_instance(optm_cfg),
                                    criterion=self._get_instance(loss_cfg))

    def _build_train_dataset(self, dataset_cfg):
        self._train_dataset = self._get_instance(dataset_cfg)

    def _build_val_dataset(self, dataset_cfg):
        if dataset_cfg:
            self._val_dataset = self._get_instance(dataset_cfg)
        else:
            self._val_dataset = None

    def _build_train_dataloader(self, dataloader_cfg):
        self._train_dataloader = DataLoader(dataset=self._train_dataset, **dataloader_cfg)

    def _build_val_dataloader(self, dataloader_cfg):
        if self._val_dataset:
            self._val_dataloader = DataLoader(dataset=self._val_dataset, **dataloader_cfg)
        else:
            self._val_dataloader = None

    def _build_logger(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger = logging.getLogger()
        handlers = []
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        handlers.append(stream_handler)
        handlers.append(logging.FileHandler(os.path.join(self.save_dir, 'log'), mode="a"))
        for handler in handlers:
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self.logger = logger

    def saveCheckpoint(self):
        torch.save({
            "state_dict": self.model.state_dict(),
        }, os.path.join(self.save_dir, f"epoch_{self.epoch}.pth"))

    @staticmethod
    def _get_instance(cfg):
        assert isinstance(cfg, dict)
        assert "name" in cfg
        return eval(cfg.pop("name"))(**cfg)
