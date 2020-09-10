import torch
import torch.nn as nn


__all__ = [
    'ShuffleNetV2', 'shufflenetv2_025', 'shufflenetv2_05', 'shufflenetv2_1',
    'shufflenetv2_15', 'shufflenetv2_2'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    # (24, 116, 2) # (116, 232, 2)
    # (116, 116, 1)
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        # 是否为下采样模块,stride = 1, stride = 2
        self.stride = stride

        # 分支通道数
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    #   [4, 8, 4], [24, 116, 232, 464, 1024]
    def __init__(self, stages_repeats, stages_out_channels,
                 inverted_residual=InvertedResidual, **kwargs):
        super(ShuffleNetV2, self).__init__()

        input_channels = kwargs.pop("input_channels", 3)
        num_classes = kwargs.pop("num_classes", 1000)

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        # 24
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #  input_channels: 24
        #     stage_names: stage2 stage3 stage4
        #         repeats: 4 8 4
        # output_channels: [116, 232, 464, 1024]

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            # (24, 116, 2)
            # (116, 232, 2)
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                # (116, 116, 1)
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            # 24 = 116
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def shufflenetv2_025(**kwargs):

    return ShuffleNetV2([4, 8, 4], [24, 24, 56, 96, 1024], **kwargs)


def shufflenetv2_05(**kwargs):

    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenetv2_1(**kwargs):

    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenetv2_15(**kwargs):

    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenetv2_2(**kwargs):

    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
