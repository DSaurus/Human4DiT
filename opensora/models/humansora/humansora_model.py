import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.svd.unet import UNetSpatioTemporalConditionModel
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint

from .models.human4dit import Human4DiT
from .models.guidance_encoder import GuidanceEncoder
from .models.guidance_encoder import zero_module    
from .models.transformer_temporal import TemporalAttention

import os.path as osp

def setup_guidance_encoder(guidance_types, guidance_encoder_kwargs, dtype):
    guidance_encoder_group = dict()

    for guidance_type in guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=guidance_encoder_kwargs.block_out_channels,
        ).to(dtype=dtype)

    return guidance_encoder_group

@MODELS.register_module()
class HumanSORA_m(nn.Module):
    def __init__(
        self,
        base_model_path,
        guidance_types,
        guidance_encoder_kwargs,
        stage="spatial", # or temporal
        dtype=torch.float32
    ):
        super().__init__()
        self.stage = stage
        
        human4dit = Human4DiT.from_pretrained_config(
            base_model_path,
        ).to(dtype=dtype)

        guidance_encoder_group = setup_guidance_encoder(guidance_types, guidance_encoder_kwargs, dtype)

        human4dit.enable_xformers_memory_efficient_attention()
        human4dit.enable_gradient_checkpointing()

        self.human4dit = human4dit
        self.guidance_types = []
        self.guidance_input_channels = []

        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)
        
        self.conv_out = zero_module(
            nn.Conv2d(
                4,
                320,
                kernel_size=1,
                padding=0,
            )
        )

        self.dtype = dtype
        self.disable_meff_temporal_attention()

    def disable_meff_temporal_attention(
        self, valid=False, attention_op=None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if isinstance(module, TemporalAttention) and hasattr(module, "set_use_memory_efficient_attention_xformers"):
                # print(module)
                module.set_use_memory_efficient_attention_xformers(False, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module,  torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, 
        x,
        timestep,
        ref_image_latents,
        clip_image_embeds,
        multi_guidance_cond,
        need_reference=True,
        guidance_feat=None,
        camera_cond=None,
    ):
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        ref_image_latents = ref_image_latents.to(self.dtype)
        # print(ref_image_latents.shape)  # torch.Size([1, 4, 96, 96])
        clip_image_embeds = clip_image_embeds.to(self.dtype)
        # print(clip_image_embeds.shape)  # torch.Size([1, 1, 1024])
        # exit(0)
        multi_guidance_cond = multi_guidance_cond.to(self.dtype)
        if camera_cond is None:
            camera_cond = torch.eye(3, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        camera_cond = camera_cond.to(self.dtype)
        if guidance_feat is not None:
            guidance_feat = guidance_feat.to(self.dtype)
    
        if guidance_feat is None:
            guidance_cond_group = torch.split(
                multi_guidance_cond, self.guidance_input_channels, dim=1
            )
            guidance_feat_lst = []
            for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
                guidance_encoder = getattr(
                    self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
                )
                guidance_feat = guidance_encoder(guidance_cond)
                guidance_feat_lst += [guidance_feat]

            guidance_feat = torch.stack(guidance_feat_lst, dim=0).mean(0)

        # print(clip_image_embeds.shape) # 1 1 1024
        
        if len(x.shape) == 6:
            # ref_image_latents = self.conv_out(ref_image_latents) # B C H W
            ref_image_latents = ref_image_latents.unsqueeze(2).unsqueeze(2)
        else:    
            # ref_image_latents = self.conv_out(ref_image_latents).unsqueeze(2).repeat(1, 1, x.shape[2], 1, 1)
            ref_image_latents = ref_image_latents.unsqueeze(2)
        model_pred = self.human4dit(
            x,
            timestep,
            guidance_feat=guidance_feat,
            ref_feat=ref_image_latents,
            encoder_hidden_states=clip_image_embeds.repeat(1, 1, 2),
            camera_cond=camera_cond,
        ).sample

        model_pred = model_pred.to(torch.float32)
        guidance_feat = guidance_feat.to(torch.float32)
        return model_pred, guidance_feat

    def clear(self):
        pass

    def fix_motion(self):
        pass

    def fix_unet(self):
        pass
    
    def fix_params(self):
        if self.stage == "spatial":
            return self.fix_motion()
        elif self.stage == "temporal":
            return self.fix_unet()
        else:
            raise ValueError(f"Invalid stage: {self.stage}")


@MODELS.register_module("HumanSORA")
def humansora_model(from_pretrained=None, **kwargs):
    model = HumanSORA_m(**kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)

    return model
