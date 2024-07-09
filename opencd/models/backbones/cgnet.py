"""
C. Han, C. Wu, H. Guo, M. Hu, J. Li and H. Chen,
"Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,"
in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
vol. 16, pp. 8395-8407, 2023, doi: 10.1109/JSTARS.2023.3310208.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from opencd.registry import MODELS


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = torch.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


@MODELS.register_module()
class CGNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg16_bn = models.vgg16_bn(pretrained=pretrained)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

        self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)

        self.decoder = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1))

        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1))

        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)

    def forward(self, x1, x2):
        size = x1.size()[2:]
        layer1_pre = self.inc(x1)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(x2)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)
        layer2 = torch.cat((layer2_B,layer2_A),dim=1)
        layer3 = torch.cat((layer3_B,layer3_A),dim=1)
        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse = layer4_1

        change_map = self.decoder(feature_fuse)

        layer4 = self.cgm_4(layer4, change_map)
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))
        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3),layer2],1))
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)

        final_map = self.decoder_final(layer1)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return (change_map, final_map)
