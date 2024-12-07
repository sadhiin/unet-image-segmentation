""" Full assembly of the parts to form the complete network """

import torch.utils
import torch.utils.checkpoint
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.2, use_batchnorm=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # Modified initialization with batch norm and dropout
        self.inc = (DoubleConv(n_channels, 64, use_batchnorm=use_batchnorm))
        self.down1 = (Down(64, 128, use_batchnorm=use_batchnorm))
        self.down2 = (Down(128, 256, use_batchnorm=use_batchnorm))
        self.down3 = (Down(256, 512, use_batchnorm=use_batchnorm))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, use_batchnorm=use_batchnorm))

        # Dropout layers
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Up-sampling path with batch norm
        self.up1 = (Up(1024, 512 // factor, bilinear, use_batchnorm=use_batchnorm))
        self.up2 = (Up(512, 256 // factor, bilinear, use_batchnorm=use_batchnorm))
        self.up3 = (Up(256, 128 // factor, bilinear, use_batchnorm=use_batchnorm))
        self.up4 = (Up(128, 64, bilinear, use_batchnorm=use_batchnorm))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.dropout(self.down1(x1))
        x3 = self.dropout(self.down2(x2))
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout(self.down4(x4))

        x = self.dropout(self.up1(x5, x4))
        x = self.dropout(self.up2(x, x3))
        x = self.dropout(self.up3(x, x2))
        x = self.dropout(self.up4(x, x1))
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=1)
    print(model)
