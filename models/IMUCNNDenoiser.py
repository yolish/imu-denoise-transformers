"""
IMUTransformer model
"""

import torch
from torch import nn

class IMUCNNDenoiser(nn.Module):
    def __init__(self, config, c0=16, dropout=0.1, ks=[7, 7, 7, 7], ds=[4, 4, 4], momentum=0.1 ):
        # The cnn archtecture is taken from: Brosard 2020
        self.in_dim = config.get("input_dim")
        self.out_dim = self.in_dim
        # channel dimension
        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        # kernel dimension (odd number)
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]
        # padding
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
        # nets
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)),  # padding at start
            torch.nn.Conv1d(self.in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0 * d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0 * d1 * d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, self.out_dim, 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)),  # no padding at end
        )

        def forward(self, data):
            orig_src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels

            # Estimate the noise residuals with convolutional backbone
            minus_noise = self.cnn(orig_src)
            # remove the noise
            clean_src = orig_src + minus_noise

            return clean_src  # N x S x C