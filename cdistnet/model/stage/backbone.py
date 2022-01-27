'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.autograd import Variable

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class BasicBlock2(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=False):
#         super().__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         if downsample:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(
#                     inplanes, planes * self.expansion, 1, stride, bias=False),
#                 nn.BatchNorm2d(planes * self.expansion),
#             )
#         else:
#             self.downsample = nn.Sequential()
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

class ConvBnRelu(nn.Module):
    # adapt padding for kernel_size change
    def __init__(self, in_channels, out_channels, kernel_size, conv = nn.Conv2d,stride=2, inplace=True):
        super().__init__()
        p_size = [int(k//2) for k in kernel_size]
        # p_size = int(kernel_size//2)
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ConvBlock(nn.Module):
    def __init__(self, in_chan,out_chan,kernel_size,downsample=True,conv=nn.Conv2d):
        super().__init__()
        if downsample == True:
            h_dim = in_chan//2 if in_chan>64 else 64
        else:
            h_dim = out_chan
        self.block = nn.Sequential(
            ConvBnRelu(in_chan,h_dim,kernel_size,conv=conv,stride=2),
            ConvBnRelu(h_dim,h_dim,3,conv=conv,stride=1),
            ConvBnRelu(h_dim,h_dim,3,conv=conv,stride=1),
        )
        if h_dim != out_chan:
            self.block.add_module('down_sample',ConvBnRelu(h_dim,out_chan,1,stride=1))

    def forward(self, x):
        x = self.block(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NRTRModalityTransform(nn.Module):

    def __init__(self, input_channels=3, input_height=32):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_1 = nn.ReLU(True)
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,1),
            stride=(2,1),
            padding=(1,0))
        self.relu_2 = nn.ReLU(True)
        self.bn_2 = nn.BatchNorm2d(64)

        feat_height = input_height // 4

        self.linear = nn.Linear(512, 512)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        n, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(n, w, h * c)
        x = self.linear(x)
        # print(x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)
        return x

class ResNet31OCR(nn.Module):
    """Implement ResNet backbone for text recognition, modified from
      `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        base_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        stage4_pool_cfg (dict): Dictionary to construct and configure
            pooling layer in stage 4.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 base_channels=3,
                 layers=[1, 2, 5, 3],
                 channels=[64, 128, 256, 256, 512, 512, 512],
                 out_indices=None,
                 stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
                 last_stage_pool=False):
        super().__init__()
        assert isinstance(base_channels, int)
        # assert utils.is_type_list(layers, int)
        # assert utils.is_type_list(channels, int)
        assert out_indices is None or (isinstance(out_indices, list)
                                       or isinstance(out_indices, tuple))
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool

        # conv 1 (Conv, Conv)
        self.conv1_1 = nn.Conv2d(
            base_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU(inplace=True)

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(
            channels[2], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(
            channels[3], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU(inplace=True)

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2d(padding=0, ceil_mode=True, **stage4_pool_cfg)
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(
            channels[4], channels[4], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU(inplace=True)

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True)  # 1/16
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(
            channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU(inplace=True)
        self.convbnrelu = ConvBnRelu(in_channels=512, out_channels=512, kernel_size=(1,3), stride=(1,2))

    def init_weights(self, pretrained=None):
        # initialize weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        bias=False),
                    nn.BatchNorm2d(output_channels),
                )
            layers.append(
                BasicBlock(
                    input_channels, output_channels, downsample=downsample))
            input_channels = output_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        outs = []
        for i in range(4):
            layer_index = i + 2
            pool_layer = getattr(self, f'pool{layer_index}')
            block_layer = getattr(self, f'block{layer_index}')
            conv_layer = getattr(self, f'conv{layer_index}')
            bn_layer = getattr(self, f'bn{layer_index}')
            relu_layer = getattr(self, f'relu{layer_index}')

            if pool_layer is not None:
                x = pool_layer(x)
            x = block_layer(x)
            x = conv_layer(x)
            x = bn_layer(x)
            x = relu_layer(x)

            # outs.append(x)
        x = self.convbnrelu(x)
        x = rearrange(x, 'b c h w -> b (w h) c')
        # if self.out_indices is not None:
        #     return tuple([outs[i] for i in self.out_indices])
        # print(x.shape)
        return x

class ABI_ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ABI_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.convbnrelu = ConvBnRelu(in_channels=512, out_channels=512, kernel_size=(3,3),stride=(2,2))
        # self.convbnrelu2 = ConvBnRelu(in_channels=512, out_channels=512, kernel_size=3,stride=2)
        # self.convbnrelu2 = ConvBnRelu(in_channels=1024, out_channels=256, kernel_size=3)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)#stride = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.convbnrelu(x)
        # x = self.convbnrelu2(x)
        x = rearrange(x, 'b c h w -> b (w h) c')
        # print(x.shape)
        return x


def ResNet45():
    return ABI_ResNet(BasicBlock, [3, 4, 6, 6, 3])
def ResNet31():
    return ResNet31OCR()
def MTB_nrtr():
    return NRTRModalityTransform()


def test():
    pass
    # net = FPN101()
    # fms = net(Variable(torch.randn(1,3,600,900)))
    # for fm in fms:
    #     print(fm.size())
    # layer1 = _make_layer(block=Bottleneck,  planes=64, num_blocks=2, stride=2)
    # layer1(torch.randn(1,64,600,900))
    # layer2 = _make_layer(Bottleneck, 128, 2, stride=2)