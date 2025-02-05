# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
from asteroid.masknn.convolutional import TDConvNet


class Net(nn.Module):
    def __init__(self, num_mics, encoder_ks, encoder_dim, encoder_stride, embed_dim, 
                 n_blocks=8, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3):
        super(Net, self).__init__()
        # Encoder for auxiliary network
        self.encoder = nn.Conv1d(num_mics, encoder_dim, encoder_ks, encoder_stride)
        
        # Auxiliary network
        self.convnet = TDConvNet(encoder_dim,
                                n_src=1,
                                out_chan=embed_dim,
                                n_blocks=n_blocks,
                                n_repeats=1,
                                bn_chan=bn_chan,
                                hid_chan=hid_chan,
                                skip_chan=skip_chan,
                                conv_kernel_size=conv_kernel_size,
                                norm_type='gLN',
                                mask_act='linear',
                                causal=False)

    def forward(self, enrollment: torch.Tensor):
        """
            x: [B, E, T]
        """
        
        x = self.encoder(enrollment)

        x = self.convnet(x)

        emb = torch.mean(x, dim = -1) # [B, E]

        return emb
    