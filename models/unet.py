"""Taken on https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        bilinear=True,
        do_activation=False,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.do_activation = do_activation

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, do_activation=None):
        act = self.do_activation if do_activation is None else do_activation

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if act:
            return F.softmax(logits, dim=1, )
        else:
            return logits


class UNetCombine(nn.Module):

    def __init__(
        self,
        unet1,
        unet2
    ):
        super().__init__()
        self.unet1 = unet1
        self.unet2 = unet2

    def forward(self, *args, **kwargs):
        ot1 = self.unet1(*args, **kwargs)
        preds1 = ot1.argmax(1)
        preds2 = self.unet2(*args, **kwargs).argmax(1)

        output = torch.zeros([ot1.shape[0], 3] + list(ot1.shape[2:]))
        output[:, 1, ...][preds1 == 1] = 1
        output[:, 2, ...][preds2 == 1] = 1

        return output
