# -*-coding:utf-8-*-
#
#    @header __init__.py
#    @author  CaoZhihui
#    @date    2020/8/19
#    @abstract:
#
from .mobilenet_v1 import (
    MobileNet, mobilenetv1_025, mobilenetv1_05, mobilenetv1_075, mobilenetv1_1, mobilenetv1_2)
from .mobilenet_v2 import (
    MobileNetV2, mobilenetv2_lv, mobilenetv2_025, mobilenetv2_05, mobilenetv2_075,
    mobilenetv2_1, mobilenetv2_2)
from .shufflenet_v2 import (
    ShuffleNetV2, shufflenetv2_025, shufflenetv2_05, shufflenetv2_1, shufflenetv2_15, shufflenetv2_2)
from .resnet import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152)
from .densenet import (DenseNet, densenet121, densenet161, densenet169, densenet201)


__all__ = [
    # MobileNet v1
    "MobileNet", "mobilenetv1_025", "mobilenetv1_05", "mobilenetv1_075", "mobilenetv1_1", "mobilenetv1_2",

    # MobileNet v2
    "MobileNetV2", "mobilenetv2_lv", "mobilenetv2_025", "mobilenetv2_05", "mobilenetv2_075",
    "mobilenetv2_1", "mobilenetv2_2",

    # ShuffleNet v2
    'ShuffleNetV2', 'shufflenetv2_025', 'shufflenetv2_05', 'shufflenetv2_1',
    'shufflenetv2_15', 'shufflenetv2_2',

    # ResNet
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',

    # DenseNet
    'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
]



