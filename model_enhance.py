# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:56:40 2024

@author: pio-r
"""


import torch
import torch.nn as nn
from torch.nn import functional as F

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)




class SuperRes(nn.Module):
    def __init__(self, c_in=1, c_out=1, channel_mult=64, device='cuda'):
        super(SuperRes, self).__init__()
        
        # Start
        self.conv1 = nn.Conv2d(c_in, channel_mult, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        
        # Res blocks
        self.resblock1 = Resblock(channel_mult, channel_mult*8)
        self.resblock2 = Resblock(channel_mult*8, channel_mult*16)
        self.resblock3 = Resblock(channel_mult*16, channel_mult*32)
        self.resblock4 = Resblock(channel_mult*32, channel_mult*16)
        self.resblock5 = Resblock(channel_mult*16, channel_mult*8)
        self.resblock6 = Resblock(channel_mult*8, channel_mult)
        
        # Bottleneck
        self.conv2 = nn.Conv2d(channel_mult, channel_mult, kernel_size=3, padding=1, bias=False)
        self.bnorm = nn.BatchNorm2d(channel_mult)
        
        # Upsampling
        
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv3 = nn.Conv2d(channel_mult, channel_mult, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.outpc = nn.Conv2d(channel_mult, c_out, kernel_size=1)
        
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        
        x2 = self.resblock1(x1)
        x3 = self.resblock2(x2)
        x4 = self.resblock3(x3)
        x5 = self.resblock4(x4)
        x6 = self.resblock5(x5)
        x7 = self.resblock6(x6)
        
        x8 = self.conv2(x7)
        x9 = self.bnorm(x8)
        
        x10 = x9 + x1
        
        x11 = self.up(x10)
        x12 = self.conv3(x11)
        x13 = self.relu2(x12)
        
        return self.outpc(x13)
        



class ProgressiveSR(nn.Module):
    def __init__(self, bicubic_512, bicubic_1024, c_in=1, c_out=1, channel_mult=64, device='cuda'):
        super(ProgressiveSR, self).__init__()
        
        self.bicubic_512 = bicubic_512
        self.bicubic_1024 = bicubic_1024
        
        
        self.conv1 = nn.Conv2d(c_in, channel_mult, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channel_mult)
        self.conv2 = nn.Conv2d(channel_mult, channel_mult*8, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channel_mult*8)
        self.conv3 = nn.Conv2d(channel_mult*8, channel_mult*16, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(1, channel_mult*16)
        self.conv4 = nn.Conv2d(channel_mult*16, channel_mult*32, kernel_size=3, padding=1, bias=False)
        self.norm4 = nn.GroupNorm(1, channel_mult*32)
        
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5 = nn.Conv2d(channel_mult*32, channel_mult*16, kernel_size=3, padding=1, bias=False)
        self.norm5 = nn.GroupNorm(1, channel_mult*16)
        self.conv6 = nn.Conv2d(channel_mult*16, channel_mult*8, kernel_size=3, padding=1, bias=False)
        self.norm6 = nn.GroupNorm(1, channel_mult*8)
        self.conv7 = nn.Conv2d(channel_mult*8, channel_mult*8, kernel_size=3, padding=1, bias=False)
        self.norm7 = nn.GroupNorm(1, channel_mult*8)
        self.conv8 = nn.Conv2d(channel_mult*8, channel_mult*4, kernel_size=3, padding=1, bias=False)
        self.norm8 = nn.GroupNorm(1, channel_mult*4)
        
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc = nn.Conv2d(channel_mult*4, c_out, kernel_size=3, padding=1, bias=False)
        
        
    def forward(self, x):
        
        
        x_512 = self.bicubic_512(x)
        x_1024 = self.bicubic_1024(x)
        
        x1 = self.norm1(self.conv1(x))
        x2 = self.norm2(self.conv2(x1))
        x3 = self.norm3(self.conv3(x2))
        x4 = self.norm4(self.conv4(x3))
        
        x5 = self.up(x4)
        x5 = x5 + x_512
        x6 = self.norm5(self.conv5(x5))
        x7 = self.norm6(self.conv6(x6))
        x8 = self.norm7(self.conv7(x7))
        x9 = self.norm8(self.conv8(x8))
        
        x10 = self.up1(x9)
        x10 = x10 + x_1024
        
        return self.outc(x10)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        