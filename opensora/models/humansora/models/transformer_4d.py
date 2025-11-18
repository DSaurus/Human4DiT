import torch
from torch import nn

from .transformer_image import TransformerImageModel
from .resnet import Downsample4D, ResnetBlock4D, Upsample4D
from .transformer_temporal import TransformerTemporal
from .transformer_multiview import TransformerMultiviewModel

class Transformer4D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            output_scale_factor=1.0,
            cross_attention_dim=1280,
            dual_cross_attention=False,
            use_linear_projection=False,
            upcast_attention=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock4D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        mv_attentions = []
        temporal_attentions = []

        for _ in range(transformer_layers_per_block):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                TransformerImageModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            mv_attentions.append(
                TransformerMultiviewModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            temporal_attentions.append(
                TransformerTemporal(
                    num_attention_heads=8,
                    attention_head_dim=in_channels // 8,
                    in_channels=in_channels,
                    cross_attention_dim=None,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.mv_attentions = nn.ModuleList(mv_attentions)
        self.temporal_attentions = nn.ModuleList(temporal_attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None,
                cross_attention_kwargs=None, enable_temporal_attentions: bool = True, ref_feat=None):
        
        # print(hidden_states.shape)  #torch.Size([1, 1280, 1, 24, 24, 24])
        # print(temb.shape)   # torch.Size([1, 1280])
        # print(ref_feat.shape)    torch.Size([1, 1280, 1, 1, 24, 24])
        # exit(0)
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, mv_att, t_att in zip(self.attentions, self.mv_attentions, self.temporal_attentions):
            if self.training and self.gradient_checkpointing:
                
                # print(self.gradient_checkpointing)
                def create_custom_forward(module, return_dict=None, ref_feat=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if ref_feat is not None:
                                return module(*inputs, return_dict=return_dict, ref_feat=ref_feat)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    use_reentrant=False
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(mv_att, return_dict=False, ref_feat=ref_feat),
                                                                    hidden_states, encoder_hidden_states,
                                                                    use_reentrant=False)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(t_att), hidden_states, encoder_hidden_states, use_reentrant=False)
            else:
                # print(hidden_states.shape)
                # exit(0)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = mv_att(hidden_states, encoder_hidden_states=encoder_hidden_states, ref_feat=ref_feat).sample
                hidden_states = t_att(hidden_states, encoder_hidden_states=encoder_hidden_states)

        return hidden_states

    def temporal_parameters(self) -> list:
        return []