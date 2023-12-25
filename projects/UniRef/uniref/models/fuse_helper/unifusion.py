from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import Mlp

from .fuse_helper import UniMultiHeadAttention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class UniFusion(torch.nn.Module):
    """
    Early Fusion Module, 
    Unified attention for both v and l.
    """

    def __init__(self, cfg):
        super(UniFusion, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        # early fusion module
        print("early fusion: ON")
        # attn
        self.norm_v = nn.LayerNorm(self.img_dim)
        self.norm_l = nn.LayerNorm(self.lang_dim)
        self.attn = UniMultiHeadAttention(
                        q_dim=self.img_dim,  # 256
                        k_dim=self.lang_dim, # 256
                        head_dim=self.head_dim, # 256
                        num_heads=self.n_head, # 8
                        dropout=0.1,
                        cfg=cfg
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.img_dim, 3 * self.img_dim, bias=True)
        )
        # mlp  TODO: whether need mlp
        # self.norm_mlp = nn.LayerNorm(self.img_dim)
        # mlp_hidden_dim = int(self.img_dim * 4.0)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=self.img_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        # adaLN
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(self.img_dim, 6 * self.img_dim, bias=True)
        # )

        # zero init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def init_configs(self, cfg):
        # common params
        self.img_dim = cfg.MODEL.DDETRS.HIDDEN_DIM  # 256
        self.lang_dim = cfg.MODEL.DDETRS.HIDDEN_DIM # 256, which has been processed 

        # mha params
        self.n_head = cfg.MODEL.DDETRS.FUSE_CONFIG.NHEADS 
        self.head_dim = cfg.MODEL.DDETRS.FUSE_CONFIG.HEAD_DIM # 256 by default


    def forward(self, srcs, lang_dict_features=None, mask_dict_features=None):
        # src (list[Tensor]): [b, c, hi, wi] -> [b, hiwi, c] -> [b, c, hi, wi]
        # lang_dict_features (dict)
        #       "ref_embeds" (Tensor): [b, c], as condition
        #       "refs" (Tensor): [b, seq, c], "ref_values" (Tensor): [b, seq, c], "masks" (BoolTensor): [b, seq], padding locations are True
        # mask_dict_features (dict)
        #       "ref_embeds" (list[Tensor]):
        #       "refs" (list[Tensor]), "ref_values" (list[Tensor]), "mask" (list[BoolTensor]), padding locations are True

        assert lang_dict_features is not None or mask_dict_features is not None

        new_srcs = []

        # for each level
        # src: q, ref: k, ref_values: v
        for i, src in enumerate(srcs):
            n, c, h, w = src.shape
            src = src.flatten(-2).permute(0, 2, 1)  # [b, hiwi, c]

            # attn
            if lang_dict_features is not None:
                # cross-attention
                ref, ref_value, mask = lang_dict_features["refs"], lang_dict_features["ref_values"], lang_dict_features["masks"]
                mask = (~mask).to(torch.long)   # reverse mask to valid locations are 1
                src_l = self.attn(self.norm_v(src), self.norm_l(ref), ref_value, attention_mask=mask)
                # shift, scale
                lang_c = lang_dict_features["ref_embeds"] 
                shift_msa_l, scale_msa_l, gate_msa_l = self.adaLN_modulation(lang_c).chunk(3, dim=1)
                src_l = gate_msa_l.unsqueeze(1) * modulate(src_l, shift_msa_l, scale_msa_l)
            if mask_dict_features is not None:
                # cross-attention
                ref, ref_value, mask = mask_dict_features["refs"][i], mask_dict_features["ref_values"][i], mask_dict_features["masks"][i]
                mask = (~mask).to(torch.long)   # reverse mask to valid locations are 1
                src_m = self.attn(self.norm_v(src), self.norm_l(ref), ref_value, attention_mask=mask)
                # shift, scale
                mask_c = mask_dict_features["ref_embeds"][i]
                shift_msa_m, scale_msa_m, gate_msa_m = self.adaLN_modulation(mask_c).chunk(3, dim=1)
                src_m = gate_msa_m.unsqueeze(1) * modulate(src_m, shift_msa_m, scale_msa_m)

            # add
            if lang_dict_features is not None and mask_dict_features is not None:
                src = src + src_l + src_m
            elif lang_dict_features is not None:
                src = src + src_l
            elif mask_dict_features is not None:
                src = src + src_m

            # # mlp
            # if lang_dict_features is not None:
            #     # mlp
            #     src_l = self.mlp(self.norm_mlp(src))
            #     # shift, scale
            #     src_l = gate_mlp_l.unsqueeze(1) * modulate(src_l, shift_mlp_l, scale_mlp_l)
            # if mask_dict_features is not None:
            #     # mlp
            #     src_m = self.mlp(self.norm_mlp(src))
            #     # shift, scale
            #     src_m = gate_mlp_m.unsqueeze(1) * modulate(src_m, shift_mlp_m, scale_mlp_m)

            # # add
            # if lang_dict_features is not None and mask_dict_features is not None:
            #     src = src + src_l + src_m
            # elif lang_dict_features is not None:
            #     src = src + src_l
            # elif mask_dict_features is not None:
            #     src = src + src_m

            # reshape
            src = src.permute(0, 2, 1).reshape(n, c, h, w)
            new_srcs.append(src)
        return new_srcs

