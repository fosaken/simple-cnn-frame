from __future__ import division

import torch.nn as nn
import torch
import math

__all__ = [
    "MobileNetV2", "mobilenetv2_lv", "mobilenetv2_025", "mobilenetv2_05", "mobilenetv2_075",
    "mobilenetv2_1", "mobilenetv2_2"]

INTERVERTED_RESIDUAL_SETTING = {
    "ori": [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]],
    "lv": [
        # t, c, n, s
        [1, 8, 1, 1],
        [6, 16, 2, 2],
        [6, 16, 3, 2],
        [6, 24, 4, 2],
        [6, 32, 3, 1],
        [6, 64, 3, 2],
        [6, 128, 1, 1]],
    "tiny": [
        # t, c, n, s
        [1, 8, 1, 1],
        [6, 16, 2, 2],
        [6, 16, 3, 2],
        [6, 24, 4, 2],
        [6, 32, 3, 1],
        [6, 64, 3, 2],
        [6, 128, 1, 1]]
}
INPUT_CHANNELS = {"ori": 32, "lv": 8, "tiny": 8}
LAST_CHANNEL = {"ori": 1280, "lv": 256, "tiny": 256}


def conv_bn(inp, oup, stride, activation="relu"):
    if activation.lower() == "relu":
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    elif activation.lower() == "prelu":
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.PReLU(),
        )
    else:
        raise NotImplementedError


def conv_1x1_bn(inp, oup, activation="relu"):
    if activation.lower() == "relu":
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    elif activation.lower() == "prelu":
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.PReLU(),
        )
    else:
        raise NotImplementedError


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation='relu'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if activation.lower() == "relu":
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif activation.lower() == "prelu":
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.PReLU(),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        else:
            if activation.lower() == "relu":
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif activation.lower() == "prelu":
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.PReLU(),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.PReLU(),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                raise NotImplementedError

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNetV2, self).__init__()

        input_channel = kwargs.pop("input_channels", 3)
        num_classes = kwargs.pop("num_classes", 1000)
        widen_factor = kwargs.pop("widen_factor", 1.0)

        mode = kwargs.pop("mode", "ori").lower()
        assert mode in {"ori", "lv", "tiny"}

        # TODO: add new activation
        activation = kwargs.pop("activation", "relu").lower()
        assert activation in {"relu", "prelu"}

        interverted_residual_setting = INTERVERTED_RESIDUAL_SETTING[mode]
        input_channels = INPUT_CHANNELS[mode]
        last_channel = LAST_CHANNEL[mode]

        block = InvertedResidual

        # building first layer
        # assert input_size % 32 == 0
        input_channels = int(input_channels * widen_factor)
        self.last_channel = int(last_channel * widen_factor) if widen_factor > 1.0 else last_channel
        self.features = [conv_bn(input_channel, input_channels, 2, activation=activation)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channels = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channels, output_channels, s, expand_ratio=t, activation=activation))
                else:
                    self.features.append(block(input_channels, output_channels, 1, expand_ratio=t, activation=activation))
                input_channels = output_channels
        # building last several layers
        self.features.append(conv_1x1_bn(input_channels, self.last_channel, activation=activation))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        self.classifier = nn.Linear(self.last_channel, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = x.mean(3).mean(2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_025(**kwargs):
    return MobileNetV2(widen_factor=0.25, **kwargs)


def mobilenetv2_05(**kwargs):
    return MobileNetV2(widen_factor=0.5, **kwargs)


def mobilenetv2_075(**kwargs):
    return MobileNetV2(widen_factor=0.75, **kwargs)


def mobilenetv2_1(**kwargs):
    return MobileNetV2(widen_factor=1.0, **kwargs)


def mobilenetv2_2(**kwargs):
    return MobileNetV2(widen_factor=2.0, **kwargs)


def mobilenetv2_lv(**kwargs):
    return MobileNetV2(mode="lv", **kwargs)
