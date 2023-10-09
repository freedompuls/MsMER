import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImageRotaryEmbed, ImgPosEnc
import matplotlib.pyplot as plt
from pylab import *




class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim: int, channel_up: int, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.bn1 = nn.BatchNorm2d(96)
        # self.pwconv1 = nn.Linear(dim, 96)  # pointwise/1x1 convs, implemented with linear layers
        self.conv1 = nn.Conv2d(dim, 96, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(96, channel_up, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_up)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(96, channel_up)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((channel_up,)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = torch.cat((shortcut, x), 1)
        return x

class _Bottleneck(nn.Module):
    def __init__(self, dim: int , channel_up:int, use_dropout=True):
        super(_Bottleneck, self).__init__()
        interChannels = 96
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(dim, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_up)
        self.conv2 = nn.Conv2d(
            interChannels, channel_up, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        if self.use_dropout:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat((x, shortcut), 1)
        return x

class downsample(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(downsample, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)
        # self.norm = LayerNorm(n_channels, eps=1e-6, data_format="channels_first")
        self.norm = nn.BatchNorm2d(n_out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.avg_pool2d(x, 2, ceil_mode=True)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 1, depths: list = None, baseWidth = 26, scale=4,
                 dims: list = None, layer_scale_init_value: float = 1e-6, use_dropout: bool = True,
                 head_init_scale: float = 1.):
        super().__init__()
        # self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem 最初的下采样部分  由卷积与layernorm构成 其中这个dims每个stage输出特征层从channel
        # self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #                      LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        # self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #                           nn.BatchNorm2d(dims[0]))
        self.conv1 = nn.Conv2d(in_chans, dims[0], kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(dims[0])

        # self.downsample1 = downsample(dims[3], dims[2], use_dropout)# 576 ->  288

        # self.downsample2 = downsample(dims[5], dims[2], use_dropout)

        # self.downsample3 = downsample(dims[6], dims[3], use_dropout)

        self.inplanes = 36
        self.baseWidth = baseWidth
        self.scale = scale
        # block堆叠次数 9 16 16
        self.layer1 = self.make_res2net_layer(dims[0], depths[0])  # 36 -> 144
        self.layer2 = self._make_dense_layer(dims[1], depths[1], 24)  # 144 -> 528
        self.downsample1 = downsample(dims[4], dims[2], use_dropout)  # 528 -> 264
        self.layer3 = self._make_dense_layer(dims[2], depths[2], 24)  # 264 -> 648
        self.downsample2 = downsample(dims[5], dims[3], use_dropout)  # 648 -> 324
        self.layer4 = self._make_dense_layer(dims[3], depths[3], 24)  # 324 -> 708

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")  # final norm layer
        # self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.post_norm = nn.BatchNorm2d(dims[-1])

    def make_res2net_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottle2neck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottle2neck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion),
            )

        layers = []
        layers.append(Bottle2neck(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * Bottle2neck.expansion
        for i in range(1, blocks):
            layers.append(Bottle2neck(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def _make_convnext_layer(self, in_channels, depth: int, channel_up: int):
        layers = []
        for _, in_channel in zip(range(depth), range(in_channels, in_channels+channel_up*depth, channel_up)):
            layers.append(Block(in_channel, channel_up))
        return nn.Sequential(*layers)

    def _make_dense_layer(self, in_channels, depth: int, channel_up: int):
        layers = []
        for _, in_channel in zip(range(depth), range(in_channels, in_channels+channel_up*depth, channel_up)):
            layers.append(_Bottleneck(in_channel, channel_up))
        return nn.Sequential(*layers)




        # dp_rates 这个列表表示的是一个等差数列  depths表示每个stage构建blocks的次数
        # 整体表示每一个block使用不同的dp_rates  并且是递增的
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0  # 表示当前的坐标
        # 构建每个stage中堆叠的block
        # 每个block输入的channel是dims[i]  drop_rate 使用的是上面构建好的 dp_rate列表 索引就是 cur+j
        # cur表示在当前这个stage 之前已经构建好的block的个数

        # for i in range(3):
        #     in_channle = dims[i]
        #     stage = nn.Sequential(
        #         *[Block(in_channle, drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
        #          for j, in_channle in zip(range(depths[i]), range(in_channle, in_channle+24*16, 24))])
        # self.stages.append(stage)
        # cur += depths[i]

    '''
        对权重进行初始化，如果权重是卷积或者全连接层 分别对权重与偏置进行初始化 
    '''
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)
    '''
        forward——features表示循环 downsample 与 stages    由i 进行表示第几个步骤  最后得到的是stage4的输出
        然后需要通过 global avg Pooling 与layer_norm 和 linear 层
        norm(x.mean([-2, -1])) 通过 mean方法 在维度 -2 -1上求一个mean均值的操作 
        -2 -1 分别表示 H W  也就是做一个global avg Pooling的操作 然后就变成 batch channle的格式了 
        (N, C, H, W) -> (N, C)
        然后就是外面的 layernorm层
    '''


    def forward(self, x: torch.Tensor, x_mask) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x_mask = x_mask[:, 0::2, 0::2]
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2, ceil_mode=True)
        x_mask = x_mask[:, 0::2, 0::2]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.downsample1(x)
        x_mask = x_mask[:, 0::2, 0::2]

        x = self.layer3(x)
        x = self.downsample2(x)
        x_mask = x_mask[:, 0::2, 0::2]

        x = self.layer4(x)
        x = self.post_norm(x)

        return x, x_mask
        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, depths: list, dims: list):
        super().__init__()

        self.model = ConvNeXt(depths=depths, dims=dims)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(dims[-1], d_model, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(d_model)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, t, d], [b, t]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")
        feature = self.norm(feature)

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)

        # flat to 1-D
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")
        return feature, mask
