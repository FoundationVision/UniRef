# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
import math
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange, repeat

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from timm.models.layers import trunc_normal_

from ..util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from detectron2.structures import Instances
from .conv_with_kaiming_uniform import conv_with_kaiming_uniform

from .fuse_helper.unifusion import UniFusion

import copy


class DDETRsegm(nn.Module):
    def __init__(self, cfg, detr,  freeze_detr=False, ota=False, rel_coord=True, new_mask_head=False, use_raft=False, 
        mask_out_stride=4, mask_out_stride_vos=1, with_lang_ref=False, with_mask_ref=False, use_early_fusion=False, 
        num_frames=1, backbone_channels=[512, 1024, 2048], use_backbone_feature=False,use_iou_branch=False
        ):
        super().__init__()
        self.debug_only = False
        self.detr = detr
        self.rel_coord = rel_coord
        print('freeze_detr:',freeze_detr)
        if freeze_detr:
            for p in self.detr.parameters():
                p.requires_grad_(False)
        self.ota = ota
        self.new_mask_head = new_mask_head
        self.use_raft = use_raft
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead

        self.use_early_fusion = use_early_fusion
        self.with_lang_ref = with_lang_ref
        self.with_mask_ref = with_mask_ref     
        self.num_frames = num_frames  # for rvos
        self.use_iou_branch = use_iou_branch
        if self.with_lang_ref or self.with_mask_ref:
            assert self.use_early_fusion, "Please make sure USE_EARLY_FUSION=True with lang/mask ref."

        # read the feature channels
        self.num_transformer_features = 3  # res3 -> res5
        self.backbone_channels = backbone_channels
        # use backbone 4x feature
        self.use_backbone_feature = use_backbone_feature
        if use_backbone_feature:
            assert len(self.backbone_channels) == 4
            # constrct conv follow mask2former
            norm = ""
            use_bias = norm == ""
            lateral_norm = get_norm(norm, hidden_dim)
            output_norm = get_norm(norm, hidden_dim)

            lateral_conv = Conv2d(
                self.backbone_channels[0], hidden_dim//32, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                hidden_dim//32, hidden_dim//32, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.lateral_conv = lateral_conv
            self.output_conv  = output_conv

            # the final conv layer
            self.mask_conv = Conv2d(
                hidden_dim//32, hidden_dim//32, kernel_size=1, stride=1, padding=0
            )

        
        # IFC-like seg
        # [N_q, num_classes]
        # self.class_embed = nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)),
        #                                  nn.Linear(int(hidden_dim/2), self.detr.num_classes))
        
        self.in_channels = hidden_dim // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.mask_out_stride = mask_out_stride
        self.mask_out_stride_vos = mask_out_stride_vos
        
        self.up_rate = 8 // self.mask_out_stride

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)

        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)
          
        if new_mask_head:
            self.mask_head = MaskHeadNew(hidden_dim, use_raft=self.use_raft, up_rate=self.up_rate)
        else:
            self.mask_head = MaskHeadSmallConv(hidden_dim, None, hidden_dim, use_raft=self.use_raft, up_rate=self.up_rate)

        # early fusion
        if self.use_early_fusion:
            self.fusion_module = UniFusion(cfg)


    def coco_forward(self, samples, gt_targets, criterion, train=False, 
                        lang_dict_features=None, mask_dict_features=None, task="detection"):
        # lang_dict_features (dict)
        #       "ref_embeds" (Tensor): [b, c], as condition
        #       "refs" (Tensor): [b, seq, c], "ref_values" (Tensor): [b, seq, c], "masks" (BoolTensor): [b, seq], padding locations are True
        # mask_dict_features (dict)
        #       "ref_embeds" (list[Tensor]):
        #       "refs" (list[Tensor]), "ref_values" (list[Tensor]), "mask" (list[BoolTensor]), padding locations are True
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        if self.debug_only:
            self.debug_data(samples, gt_targets)
        features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, (feat, pos_l) in enumerate(zip(features[-self.num_transformer_features:], pos[-self.num_transformer_features:])):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features[-self.num_transformer_features:]):
            _len_srcs = len(features[-self.num_transformer_features:])
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        # early-fusion
        if self.use_early_fusion:
            if task == "grounding":
                assert lang_dict_features is not None
                srcs = self.fusion_module(srcs, lang_dict_features=lang_dict_features)
            elif task == "fss" or task == "vos":
                assert mask_dict_features is not None
                srcs = self.fusion_module(srcs, mask_dict_features=mask_dict_features)
            elif task == "rvos":
                assert lang_dict_features is not None
                srcs = self.fusion_module(srcs, lang_dict_features=lang_dict_features, mask_dict_features=mask_dict_features)

        if (not self.detr.two_stage or self.detr.mixed_selection):
            query_embeds = self.detr.query_embed.weight
        else:
            query_embeds = None

        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.detr.transformer(srcs, masks, poses, query_embeds, mask_on=True)

        # memory: [N, \sigma(HiWi), C], from 8x -> 32x 
        # hs: [num_encoders, N, num_querries, C]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        # define a list for ious, could be empty
        outputs_ious = []

        enc_lay_num = hs.shape[0]
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            if self.use_iou_branch:
                pred_iou = self.detr.iou_head[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.use_iou_branch:
                outputs_ious.append(pred_iou)

            # IFC class_embed & mask_embed
            # class_logits = self.class_embed(hs[lvl])  # [bs, num_querries, num_classes]

            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]

            # for training & log evaluation loss
            if self.ota:
                indices, matched_ids = criterion.matcher.forward_ota(outputs_layer, gt_targets)
            else:
                indices = criterion.matcher.forward_ori(outputs_layer, gt_targets)
            indices_list.append(indices)
            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                if self.ota:
                    # num_insts.append(pred_i.sum())
                    num_insts.append(len(pred_i))   # the pred_i is indice instead of bool
                else:
                    num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)
                # import pdb;pdb.set_trace()
                orig_h, orig_w = image_sizes[i]
                orig_h = torch.as_tensor(orig_h).to(reference)
                orig_w = torch.as_tensor(orig_w).to(reference)
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                
                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[:, :2]
                ref_cur_f = ref_cur_f * scale_f[None, :]
                reference_points.append(ref_cur_f[pred_i].unsqueeze(0))

            # reference_points: [1, \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = torch.cat(mask_head_params, dim=1)

            # mask prediction
            has_mask_list = ["masks" in x.keys() for x in gt_targets]
            assert len(set(has_mask_list)) == 1 # must be "all True" or "all False"
            if has_mask_list[0]:
                extra_feature = features[0].tensors if self.use_backbone_feature else None
                outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_points, mask_head_params, num_insts,
                                                            extra_feature=extra_feature, task=task)
            else:
                # avoid unused parameters
                dummy_output = torch.sum(mask_head_params)
                for p in self.mask_head.parameters():
                    dummy_output += p.sum()
                outputs_layer['pred_masks'] = 0.0 * dummy_output
            outputs_masks.append(outputs_layer['pred_masks'])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks
        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]
        # import pdb;pdb.set_trace()
        
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        if self.use_iou_branch:
            outputs_iou = torch.stack(outputs_ious)
            outputs['pred_boxious'] = outputs_iou[-1]
        else: 
            outputs_iou = []

        if self.detr.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask, outputs_iou)

        if self.detr.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # # Retrieve the matching between the outputs of the last layer and the targets
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # indices = criterion.matcher(outputs_without_aux, gt_targets)
        loss_dict = criterion(outputs, gt_targets, indices_list)

        return outputs, loss_dict

    
    def coco_inference(self, samples, gt_targets, criterion, train=False, 
                        lang_dict_features=None, mask_dict_features=None, task="detection"):
        # if isinstance(samples, (list, torch.Tensor)):
        image_sizes = samples.image_sizes
        # import pdb;pdb.set_trace()
        if not isinstance(samples, NestedTensor):
            # size_divisibility = getattr(self.detr.backbone[0].backbone, "size_divisibility", 32)
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)

        features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, (feat, pos_l) in enumerate(zip(features[-self.num_transformer_features:], pos[-self.num_transformer_features:])):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features[-self.num_transformer_features:]):
            _len_srcs = len(features[-self.num_transformer_features:])
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        # early-fusion
        if self.use_early_fusion:
            if task == "grounding":
                assert lang_dict_features is not None
                srcs = self.fusion_module(srcs, lang_dict_features=lang_dict_features)
            elif task == "fss" or task == "vos":
                assert mask_dict_features is not None
                srcs = self.fusion_module(srcs, mask_dict_features=mask_dict_features)
            elif task == "rvos":
                assert lang_dict_features is not None
                srcs = self.fusion_module(srcs, lang_dict_features=lang_dict_features, mask_dict_features=mask_dict_features)

        if (not self.detr.two_stage or self.detr.mixed_selection):
            query_embeds = self.detr.query_embed.weight
        else:
            query_embeds = None

        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.detr.transformer(srcs, masks, poses, query_embeds, mask_on=True)

        # memory: [N, \sigma(HiWi), C]
        # hs: [num_encoders, N, num_querries, C]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        outputs_ious = []

        enc_lay_num = hs.shape[0]
        # for lvl in range(enc_lay_num):
        lvl = enc_lay_num - 1
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.detr.class_embed[lvl](hs[lvl])
        if self.use_iou_branch:
            outputs_iou = self.detr.iou_head[lvl](hs[lvl])
        tmp = self.detr.bbox_embed[lvl](hs[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.use_iou_branch:
            outputs_ious.append(outputs_iou)
            outputs_iou = torch.stack(outputs_ious)
            outputs['pred_boxious'] = outputs_iou[-1]
        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]
        # import pdb;pdb.set_trace()
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]

        # # Retrieve the matching between the outputs of the last layer and the targets
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # indices = criterion.matcher(outputs_without_aux, gt_targets)
        if train:
            loss_dict = criterion(outputs, gt_targets, indices_list)
        else:
            loss_dict = None
        
        if not train:
            outputs['reference_points'] = inter_references[-2, :, :, :2]
            dynamic_mask_head_params = self.controller(hs[-1])    # [bs, num_quries, num_params]            
            bs, num_queries, _ = dynamic_mask_head_params.shape
            num_insts = [num_queries for i in range(bs)]
            # import pdb;pdb.set_trace()
            reference_points = []
            for i, image_size_i in enumerate(image_sizes):
                orig_h, orig_w = image_size_i
                orig_h = torch.as_tensor(orig_h).to(outputs['reference_points'][i])
                orig_w = torch.as_tensor(orig_w).to(outputs['reference_points'][i])
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                ref_cur_f = outputs['reference_points'][i] * scale_f[None, :]
                reference_points.append(ref_cur_f.unsqueeze(0))
            
            # reference_points: [1, N * num_queries, 2]
            # mask_head_params: [1, N * num_queries, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = dynamic_mask_head_params.reshape(1, -1, dynamic_mask_head_params.shape[-1])
            # import pdb;pdb.set_trace()
            # mask prediction
            extra_feature = features[0].tensors if self.use_backbone_feature else None
            outputs = self.forward_mask_head_train(outputs, memory, spatial_shapes, 
                                                   reference_points, mask_head_params, num_insts,
                                                   extra_feature=extra_feature, task=task)
            # import pdb;pdb.set_trace()
            # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]
            outputs['pred_masks'] = torch.cat(outputs['pred_masks'], dim=0)

        return outputs, loss_dict


    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts, extra_feature=None, task="det"):
        bs, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0,4,1,2,3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w
        
        pred_masks = []
        # NOTE: here iframe is 1!
        for iframe in range(1):
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]

            # feats = [] # features[3], features[2], features[1]
            # for i in range(self.detr.num_feature_levels - 1, 0, -1):
            #     N, _c, _h, _w = features[i].tensors.shape
            #     feats.append(features[i].tensors.reshape(bs, self.detr.num_frames, _c, _h, _w)[:,iframe,:,:,:])
            
            # decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
            # encod_feat_f (list[tensor]): from 8x -> 32x
            if self.new_mask_head:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(encod_feat_f)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f)
                    up_masks = None
            else:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(encod_feat_f, fpns=None)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
                    up_masks = None
            # decod_feat_f = self.spatial_decoder(encod_feat_f)[0]  
            # [bs, C/32, H/8, W/8]
            # whether need 4x feature
            if self.use_backbone_feature:
                assert extra_feature is not None
                cur_fpn = self.lateral_conv(extra_feature)
                y = cur_fpn + F.interpolate(decod_feat_f, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                y = self.output_conv(y)
                decod_feat_f = self.mask_conv(y) # [B, C/32, H/4, W/4]

            ######### conv ##########
            mask_feat_stride = 4 if self.use_backbone_feature else 8
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=mask_feat_stride,
                                                        rel_coord=self.rel_coord, up_masks=up_masks, task=task)
            # mask_logits: [1, num_queries_all, H/4, W/4]


            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)  
        
        # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))
        
        outputs['pred_masks'] = output_pred_masks

        return outputs


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts, 
                                 mask_feat_stride, rel_coord=True, up_masks=None, task='det'):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        
        if rel_coord:
            instance_locations = reference_points
            # instance_locations: [1, num_insts_all, 2]
            # locations: [H*W, 2]
            # import pdb;pdb.set_trace()
            # relative_coords = locations.reshape(1, 1, H, W, 2).repeat(1,num_insts_all,1,1,1)
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            # relative_coords: [1, num_insts_all, H, W, 2]

            # # coords normalization
            # scale = torch.tensor([W, H],device=device)
            # relative_coords = relative_coords.float() / scale[None, None, None, None, :]

            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            # relative_coords: [1, num_insts_all, 2, H*W]
            # import pdb;pdb.set_trace()

            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # import pdb;pdb.set_trace()

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
       
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs + torch.sum(mask_head_params) * 0.0
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        # import pdb;pdb.set_trace()
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        # mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        if self.use_raft:
            assert up_masks is not None
            inst_idx = 0
            mask_logits_output = []
            for b, n in enumerate(num_insts):
                mask_logits_output.append(self.upsample_preds(mask_logits[inst_idx:inst_idx+n], up_masks[b:b+1]))
                inst_idx += n
            mask_logits = torch.cat(mask_logits_output, dim=0)
        else:
            # vos mask_stride = 1 else 4
            mask_stride = self.mask_out_stride_vos if task == "vos" else self.mask_out_stride
            mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / mask_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits


    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask, outputs_iou=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.use_iou_branch:
            output = [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c, 'pred_boxious': d}
                    for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1], outputs_iou)]
        else:
            output = [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]
        return output


    def upsample_preds(self, pred, mask):
        """ Upsample pred [N, 1, H/8, W/8] -> [N, 1, H, W] using convex combination """
        N, _, H, W = pred.shape
        mask = mask.view(1, 1, 9, self.up_rate, self.up_rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, 1, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, 1, self.up_rate*H, self.up_rate*W)

    def debug_data(self, samples, gt_targets):
        import numpy as np
        import copy
        import cv2
        import torch.distributed as dist
        import sys
        import time
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        for i in range(len(gt_targets)):
            image = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
            input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
            image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
            target = gt_targets[i]
            boxes = target["boxes"].cpu().numpy()
            num_inst = boxes.shape[0]
            for j in range(num_inst):
                cx, cy, w, h = boxes[j] * target["image_size"].cpu().numpy() # image size without padding
                x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
                # mask = target["masks"][j].cpu().float().numpy() # (H, W)
                # image_new = copy.deepcopy(image)
                # if mask.shape != image_new.shape[:-1]:
                #     ori_h, ori_w = mask.shape
                #     mask_new = np.zeros((image_new.shape[:-1]))
                #     mask_new[:ori_h, :ori_w] = mask
                # else:
                #     mask_new = mask
                # image_new[:, :, -1] += 128 * mask_new
                # cv2.rectangle(image_new, (x1, y1), (x2, y2), (255,0,0), thickness=2)
                # cv2.imwrite("rank_%02d_batch_%d_inst_%dimg.jpg"%(dist.get_rank(), i, j), image_new)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), thickness=2)
            cv2.imwrite("rank_%02d_batch_%d_img.jpg"%(dist.get_rank(), i), image)
            cv2.imwrite("rank_%02d_batch_%d_mask.jpg"%(dist.get_rank(), i), input_mask)
        time.sleep(5)
        sys.exit(0)
        

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, use_raft=False, up_rate=4):
        super().__init__()
        self.use_raft = use_raft
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2 # original value is 8 (compared with 4x downsampled mask, here should be 2)
        self.up_rate = up_rate
        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim//4, 3, padding=1)
        # self.gn1 = torch.nn.GroupNorm(8, dim//4)
        self.lay2 = torch.nn.Conv2d(dim//4, dim//32, 3, padding=1)
        # self.gn2 = torch.nn.GroupNorm(8, dim//32)

        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        # self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        # self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        # self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        # self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)#, bias=False)
        # self.dcn = DeformConv(inter_dims[3],inter_dims[4], 3, padding=1)
        self.jia_dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)
        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(context_dim, self.up_rate*self.up_rate*9, 1, padding=0))

    def forward(self, x, fpns):
        # enc_p3, enc_p4, enc_p5 = x
        # x = self.lay1(x)
        # x = self.gn1(x)
        # x = F.relu(x)
        # x = self.lay2(x)
        # x = self.gn2(x)
        # x = F.relu(x)

        # x (list[tenosr]):    [B, C, Hi, Wi], from 8x -> 32x
        # fpns (list[tensor]): if any, [B, C, Hi, Wi], from 32x -> 8x
        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        # fused_x = self.gn3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        # fused_x = self.gn4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        # dcn for the last layer
        # offset = self.conv_offset(x)
        # x = self.dcn(x,offset)
        fused_x = self.jia_dcn(fused_x)
        # fused_x = self.gn5(fused_x)
        fused_x_fpn = F.relu(fused_x)

        fused_x = self.lay1(fused_x_fpn)
        # fused_x = self.gn1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        # fused_x = self.gn2(fused_x)
        fused_x = F.relu(fused_x)

        if self.use_raft:
            up_masks = self.up_mask_layer(fused_x_fpn) # weights used for upsampling the coarse mask predictions
            return fused_x, up_masks
        else:
            return fused_x

class MaskHeadNew(nn.Module):
    """
    22.04.04 New mask head (as same as CondInst)
    """

    def __init__(self, in_channels, channels=128, num_convs=4, sem_loss_on=False, num_classes=80, use_raft=False, up_rate=4):
        super().__init__()

        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        self.num_outputs = 8
        self.use_raft = use_raft
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2 # original value is 8 (compared with 4x downsampled mask, here should be 2)
        self.up_rate = up_rate
        self.sem_loss_on = sem_loss_on

        self.refine = nn.ModuleList()
        for _ in range(3):
            self.refine.append(conv_block(in_channels, channels, 3))

        tower = nn.ModuleList()
        for _ in range(num_convs):
            tower.append(conv_block(channels, channels, 3))
        tower.append(nn.Conv2d(channels, max(self.num_outputs, 1), 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, self.up_rate*self.up_rate*9, 1, padding=0))

        if self.sem_loss_on:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0

            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)


    def forward(self, features):
        """gt_bitmasks_full: (bs, M, H, W), gt_classes: (bs, M)"""
        # NOTE: gt_bitmasks_full has been downsampled by 4 (to reduce latency)
        # Here CondInst uses multiple-level features (p3, p4, p5)
        # -3, -2, -1 corresponds to P3, P4, P5
        for i, f in enumerate([-3, -2, -1]):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)


        if self.use_raft:
            up_masks = self.up_mask_layer(x) # weights used for upsampling the coarse mask predictions
            return mask_feats, up_masks
        else:
            return mask_feats

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def segmentation_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5, filter_empty: bool = True
    ):

    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    if filter_empty:
        results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        mask = F.interpolate(results.pred_masks.float(), size=(output_height, output_width), mode='nearest')
        mask = mask.squeeze(1).byte()
        results.pred_masks = mask
    return results



