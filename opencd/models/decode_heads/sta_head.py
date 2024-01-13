# Copyright (c) Open-CD. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS


class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel ** -.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input

        return out


class _PAMBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    '''

    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)  # B*N*H*W*2

        local_block_cnt = 2 * self.scale * self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels ** -.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            # context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                  range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                  range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list, dim=0)
        context_locals = func(v_locals, q_locals, k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])

        return context


class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels // 8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels, key_channels, value_channels, scale, ds)


class PAM(nn.Module):
    """
        PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8

        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds)
             for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels, key_channels, value_channels, size, ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))

        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CDSA(nn.Module):
    """self attention module for change detection
    """

    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1, 2, 4, 8], ds=self.ds)
        elif self.mode == 'None':
            self.Self_Att = nn.Identity()
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


@MODELS.register_module()
class STAHead(BaseDecodeHead):
    """The Head of STANet.

    Args:
        sa_mode:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(
        self, 
        sa_mode='PAM', 
        sa_in_channels=256, 
        sa_ds=1, 
        distance_threshold=1, 
        **kwargs):
        super().__init__(input_transform='multiple_select', num_classes=1, **kwargs)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.distance_threshold = distance_threshold

        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            fpn_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        
        self.fpn_bottleneck = nn.Sequential(
            ConvModule(
                len(self.in_channels) * self.channels,
                sa_in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Dropout(0.5),
            ConvModule(
                sa_in_channels,
                sa_in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        
        self.netA = CDSA(in_c=sa_in_channels, ds=sa_ds, mode=sa_mode)
        self.calc_dist = nn.PairwiseDistance(keepdim=True)
        self.conv_seg = nn.Identity()
                
    def base_forward(self, inputs):
        fpn_outs = [
            self.fpn_convs[i](inputs[i])
            for i in range(len(self.in_channels))
        ]

        for i in range(len(self.in_channels)):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        f1 = self.base_forward(inputs1)
        f2 = self.base_forward(inputs2)
        f1, f2 = self.netA(f1, f2)

        # if you use PyTorch<=1.8, there may be some problems. 
        # see https://github.com/justchenhao/STANet/issues/85
        f1 = f1.permute(0, 2, 3, 1)
        f2 = f2.permute(0, 2, 3, 1)
        dist = self.calc_dist(f1, f2).permute(0, 3, 1, 2)

        dist = F.interpolate(dist, size=inputs[0].shape[2:], mode='bilinear', align_corners=True)

        return dist

    def predict_by_feat(self, seg_logits, batch_img_metas):
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits_copy = copy.deepcopy(seg_logits)
        seg_logits[seg_logits_copy > self.distance_threshold] = 100
        seg_logits[seg_logits_copy <= self.distance_threshold] = -100

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
