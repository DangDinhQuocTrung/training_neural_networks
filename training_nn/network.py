import torch
import torch.nn as nn
import training_nn.normalization as normalization


def conv_layer(in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=kernel_size, stride=stride, padding=padding,
        groups=groups, dilation=dilation, bias=False)


def get_norm_layer(c, h, w):
    return normalization.InstanceNorm(h, w)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes, planes, tensor_size,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1,
        norm_layer_func=get_norm_layer
    ):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(in_planes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer_func(planes, tensor_size // stride, tensor_size // stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(planes, planes, 3, padding=1)
        self.bn2 = norm_layer_func(planes, tensor_size // stride, tensor_size // stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, tensor_size,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1,
        norm_layer_func=get_norm_layer
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(in_planes, width, 1)
        self.bn1 = norm_layer_func(width, tensor_size, tensor_size)
        self.conv2 = conv_layer(width, width, 3, stride, dilation, groups, dilation)
        self.bn2 = norm_layer_func(width, tensor_size // stride, tensor_size // stride)
        self.conv3 = conv_layer(width, planes * self.expansion, 1)
        self.bn3 = norm_layer_func(planes * self.expansion, tensor_size // stride, tensor_size // stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer_func=get_norm_layer
    ):
        super(ResNet, self).__init__()

        self.in_planes = 64
        self.tensor_size = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3
        self.norm_layer_func = norm_layer_func

        self.groups = groups
        self.base_width = width_per_group

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv_layer(3, self.in_planes, 7, stride=2, padding=3)
        self.bn1 = norm_layer_func(self.in_planes, self.tensor_size // 2, self.tensor_size // 2)

        self.layer1 = self._make_layer(block, 64, layers[0], self.tensor_size // 4)
        self.layer2 = self._make_layer(
            block, 128, layers[1], self.tensor_size // 4,
            stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], self.tensor_size // 8,
            stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], self.tensor_size // 16,
            stride=2, dilate=replace_stride_with_dilation[2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck) and hasattr(m.bn3, "weight"):
                nn.init.constant_(m.bn3.weight, 0)
        return

    def _make_layer(self, block, planes, blocks, tensor_size, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_layer(self.in_planes, planes * block.expansion, 1, stride),
                self.norm_layer_func(planes * block.expansion, tensor_size // stride, tensor_size // stride),
            )

        layers = [
            block(
                self.in_planes, planes, tensor_size, stride, downsample, self.groups,
                self.base_width, previous_dilation, self.norm_layer_func),
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes, planes, tensor_size // stride, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer_func=self.norm_layer_func))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], 10)


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], 10)
