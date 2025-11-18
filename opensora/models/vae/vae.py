import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange

from opensora.registry import MODELS


@MODELS.register_module()
class VideoAutoencoderKL(nn.Module):
    def __init__(self, from_pretrained=None, micro_batch_size=None, sub_folder=None):
        super().__init__()
        if from_pretrained == "stabilityai/stable-video-diffusion-img2vid-xt":
            self.module = AutoencoderKLTemporalDecoder.from_pretrained(from_pretrained, subfolder="vae")
            self.video = True
        else:
            if sub_folder is not None:
                self.module = AutoencoderKL.from_pretrained(from_pretrained, subfolder=sub_folder)
            else:
                self.module = AutoencoderKL.from_pretrained(from_pretrained)
            self.video = False
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x):
        # x: (B, C, T, H, W)
        
        if len(x.shape) == 6:
            is_mv_video = True
            B, C, V, T, H, W = x.shape
        else:
            is_mv_video = False
            B, C, T, H, W = x.shape

        if is_mv_video:
            x = rearrange(x, "B C V T H W -> (B V T) C H W")
        else:
            x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.mode().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.mode().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        
        if is_mv_video:
            x = rearrange(x, "(B V T) C H W -> B C V T H W", B=B, V=V)
        else:
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        if len(x.shape) == 6:
            is_mv_video = True
            B, C, V, T, H, W = x.shape
        else:
            is_mv_video = False
            B, C, T, H, W = x.shape
        
        if is_mv_video:
            x = rearrange(x, "B C V T H W -> (B V T) C H W")
        else:
            x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            if self.video:
                x = self.module.decode(x / 0.18215, num_frames=T).sample
            else:
                x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        if is_mv_video:
            x = rearrange(x, "(B V T) C H W -> B C V T H W", B=B, V=V)
        else:
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size


@MODELS.register_module()
class VideoAutoencoderKLTemporalDecoder(nn.Module):
    def __init__(self, from_pretrained=None, sub_folder=None):
        super().__init__()
        self.module = AutoencoderKLTemporalDecoder.from_pretrained(from_pretrained)
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        B, _, T = x.shape[:3]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.module.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size
