import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):

        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1  = x01[:, :, :, 0::2]
        x2  = x02[:, :, :, 0::2]
        x3  = x01[:, :, :, 1::2]
        x4  = x02[:, :, :, 1::2]
        
        xll =   x1 + x2 + x3 + x4
        xlh = - x1 + x2 - x3 + x4
        xhl = - x1 - x2 + x3 + x4
        xhh =   x1 - x2 - x3 + x4

        return torch.cat((xll, xhl, xlh, xhh), 1)


class IWT(nn.Module):
    def __init__(self, cuda_flag=True):
        super().__init__()
        self.requires_grad = False
        self.cuda_flag     = cuda_flag

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()
        if self.cuda_flag: h = h.cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


# example


# DWT
# l1 = nn.Conv2d(in_ch1, out_ch1, ker_size, stride=2)
# l2 = nn.Conv2d(in_ch2, out_ch2, ker_size)
# ------>
# l1 = nn.Conv2d(in_ch, out_ch, ker_size, stride=1)
# wp = DWT()
# l2 = nn.Conv2d(in_ch2*2, out_ch, ker_size)

