import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class BasicBlock(nn.Module):   # 잔차 블럭
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=7, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False), # preseve resoltuion : 32x32
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )

        self.stage_layer = nn.Sequential(
            self._make_layer(block,  64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),  # 32x32 --> 16x16
            self._make_layer(block, 256, layers[2], stride=2),  # 16x16 --> 8x8
            self._make_layer(block, 512, layers[3], stride=1),  # preserve resolution : 8x8
            nn.AdaptiveAvgPool2d((1, 1))                      # Global Average Pooling (GAP) : 512x8x8 --> 512x1x1
        )

        self.fc_layer = nn.Sequential(
            #nn.Linear(512 * block.expansion, 2048),
            #nn.ReLU(),
            #nn.Linear(2048, 2048),
            #nn.ReLU(),
            nn.Linear(512 * block.expansion,num_classes)
        )


        # weight initaliztaion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # projection shortcut
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))   # downsample input
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)
        x = self.stage_layer(x)
        # print(f'gap shape : {x.shape}')
        x = torch.flatten(x, start_dim=1)
        # print(f'flatten shape : {x.shape}')
        x = self.fc_layer(x)

        return x



def _resnet(block, layers):
    model = ResNet(block, layers)
    return model

def resnet11():
    return _resnet( BasicBlock, [1, 1, 1, 1])

def resnet18():
    return _resnet( BasicBlock, [2, 2, 2, 2])

def resnet34():
    return _resnet( BasicBlock, [3, 4, 6, 3])

def resnet50():
    return _resnet( Bottleneck, [3, 4, 6, 3])

def resnet101():
    return _resnet( Bottleneck, [3, 4, 23, 3])