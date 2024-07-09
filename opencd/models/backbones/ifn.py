# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from opencd.registry import MODELS


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d


def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU

def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)


def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

class BasicConv(nn.Module):
    def __init__(
        self, in_ch, out_ch, 
        kernel_size, pad_mode='Zero', 
        bias='auto', norm=False, act=False, 
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = Conv1x1(in_ch, in_ch//ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch//ratio, in_ch, bias=False)

    def forward(self,x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class VGG16FeaturePicker(nn.Module):
    def __init__(self, indices=(3,8,15,22,29)):
        super().__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()
        self.indices = set(indices)

    def forward(self, x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_ch, out_ch, with_dropout=True):
    lst = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        make_norm(out_ch),
    ]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)


@MODELS.register_module()
class IFN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.encoder1 = self.encoder2 = VGG16FeaturePicker()

        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, 1)

    def forward(self, t1, t2):
        # Extract bi-temporal features
        with torch.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29,= t2_feats

        # Multi-level decoding
        x = torch.cat([t1_f_l29, t2_f_l29], dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        out1 = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = torch.cat([x, t1_f_l22, t2_f_l22], dim=1)
        x = self.ca2(x)*x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        out2 = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = torch.cat([x, t1_f_l15, t2_f_l15], dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        out3 = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = torch.cat([x, t1_f_l8, t2_f_l8], dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        out4 = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = torch.cat([x, t1_f_l3, t2_f_l3], dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        out5 = self.o5_conv4(x)

        return (out1, out2, out3, out4, out5)