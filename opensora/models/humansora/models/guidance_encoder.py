from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from .resnet import Conv4d
from .transformer_image import TransformerImageModel


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class GuidanceEncoder(ModelMixin):
    def __init__(
        self,
        guidance_embedding_channels: int,
        guidance_input_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        attention_num_heads: int = 8,
    ):
        super().__init__()
        self.guidance_input_channels = guidance_input_channels
        self.conv_in = Conv4d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.norms = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                Conv4d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.norms.append(torch.nn.GroupNorm(num_groups=4, num_channels=channel_in, eps=1e-5, affine=True))
            self.blocks.append(
                Conv4d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
            self.norms.append(torch.nn.GroupNorm(num_groups=4, num_channels=channel_out, eps=1e-5, affine=True))
            

        self.conv_out = zero_module(
            Conv4d(
                block_out_channels[-1],
                guidance_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, condition):
        embedding = self.conv_in(condition)
        embedding = F.silu(embedding)

        for norm, block in zip(self.norms, self.blocks):
            embedding = block(embedding)
            embedding = norm(embedding)
            embedding = F.silu(embedding)

        # FIXME: Temporarily only use the last attention.
        # embedding = self.attentions[-1](embedding).sample
        embedding = self.conv_out(embedding)

        return embedding
