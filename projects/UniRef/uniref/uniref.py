# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from re import template

from torch._C import Value
from detectron2.layers.shape_spec import ShapeSpec
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import copy
from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from fvcore.nn import giou_loss, smooth_l1_loss

from .backbone.masked_backbone import MaskedBackbone
from .models.deformable_detr.backbone import Joiner
from .models.deformable_detr.deformable_detr import DeformableDETR
from .models.deformable_detr.matcher import HungarianMatcher
from .models.deformable_detr.criterion import SetCriterion
from .models.deformable_detr.position_encoding import PositionEmbeddingSine
from .models.deformable_detr.deformable_transformer import DeformableTransformer
from .models.ddetrs import DDETRsegm, segmentation_postprocess
from .models.vos_helper.modules import ValueEncoderSO
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
import torchvision.ops as ops

# Language
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import BertModel, BertTokenizerFast
from transformers import CLIPTextModel, CLIPTokenizerFast
from einops import repeat
import os
from PIL import Image
from skimage import color

# for visualization
from detectron2.structures import BoxMode
import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

__all__ = ["UniRef"]


@META_ARCH_REGISTRY.register()
class UniRef(nn.Module):
    """
    Implement DDETRS
    """

    def __init__(self, cfg):
        super().__init__()
        self.global_num = 0 # for visualization
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM  # for image, is 1 
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.test_topk = cfg.TEST.DETECTIONS_PER_IMAGE

        self.num_classes = cfg.MODEL.DDETRS.NUM_CLASSES
        self.mask_stride = cfg.MODEL.DDETRS.MASK_STRIDE          # 4
        self.mask_stride_vos = cfg.MODEL.DDETRS.MASK_STRIDE_VOS  # 1
        self.match_stride = cfg.MODEL.DDETRS.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.ota = cfg.MODEL.OTA
        self.test_nms = cfg.MODEL.TEST_NMS

        self.mask_thres = cfg.MODEL.DDETRS.MASK_THRES
        self.new_mask_head = cfg.MODEL.DDETRS.NEW_MASK_HEAD
        self.use_raft = cfg.MODEL.DDETRS.USE_RAFT
        self.use_rel_coord = cfg.MODEL.DDETRS.USE_REL_COORD
        self.freeze_detr = cfg.FREEZE_DETR
        self.num_queries = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES

        # Task parameters
        self.nshot = cfg.TASK.FSS.NSHOT

        # Transformer parameters:
        hidden_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        nheads = cfg.MODEL.DDETRS.NHEADS
        dropout = cfg.MODEL.DDETRS.DROPOUT
        dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DDETRS.ENC_LAYERS
        dec_layers = cfg.MODEL.DDETRS.DEC_LAYERS
        enc_n_points = cfg.MODEL.DDETRS.ENC_N_POINTS
        dec_n_points = cfg.MODEL.DDETRS.DEC_N_POINTS
        num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DDETRS.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS
        assert two_stage_num_proposals == self.num_queries
        
        # Loss parameters:
        mask_weight = cfg.MODEL.DDETRS.MASK_WEIGHT
        dice_weight = cfg.MODEL.DDETRS.DICE_WEIGHT
        giou_weight = cfg.MODEL.DDETRS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DDETRS.L1_WEIGHT
        class_weight = cfg.MODEL.DDETRS.CLASS_WEIGHT
        mask_aux_weight = cfg.MODEL.DDETRS.MASK_AUX_WEIGHT
        deep_supervision = cfg.MODEL.DDETRS.DEEP_SUPERVISION
        focal_alpha = cfg.MODEL.DDETRS.FOCAL_ALPHA
        # cost parameters
        set_cost_class = cfg.MODEL.DDETRS.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.DDETRS.SET_COST_BOX
        set_cost_giou = cfg.MODEL.DDETRS.SET_COST_GIOU

        # Backbone
        d2_backbone = MaskedBackbone(cfg)
        # backbone out shapes, from res2 -> res5
        self.backbone_channels = d2_backbone.num_channels
        self.backbone_strides = d2_backbone.feature_strides
        # transformer in features, from res3 -> res5
        transformer_in_features = cfg.MODEL.DDETRS.IN_FEATURES 
        self.num_transformer_features = len(transformer_in_features)
        # set the backbone channels and strides, this is for transformer input_proj
        N_steps = hidden_dim // 2
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.num_channels = self.backbone_channels[-self.num_transformer_features:]  
        backbone.strides = self.backbone_strides[-self.num_transformer_features:]

        # Transformer
        transformer = DeformableTransformer(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,
        two_stage=two_stage,
        two_stage_num_proposals=two_stage_num_proposals,
        look_forward_twice=cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        use_checkpoint=cfg.MODEL.DDETRS.USE_CHECKPOINT,
        )
        
        # DETR
        model = DeformableDETR(
        cfg,
        backbone,
        transformer,
        num_classes=self.num_classes,
        num_queries=self.num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=two_stage,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        use_iou_branch=cfg.MODEL.DDETRS.USE_IOU_BRANCH)

        # DETR + Segmentation
        self.with_mask_ref = cfg.MODEL.WITH_MASK_REF
        self.with_lang_ref = cfg.MODEL.WITH_LANG_REF
        self.use_early_fusion = cfg.MODEL.USE_EARLY_FUSION
        self.use_backbone_feature = cfg.MODEL.DDETRS.USE_BACKBONE_FEATURE  # whether use backbone 4x feature for segmentation FPN
        self.use_iou_branch = cfg.MODEL.DDETRS.USE_IOU_BRANCH
        self.detr = DDETRsegm(cfg, model, freeze_detr=self.freeze_detr,  ota=self.ota, rel_coord=self.use_rel_coord,
                new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, mask_out_stride_vos=self.mask_stride_vos,
                with_lang_ref=self.with_lang_ref, with_mask_ref=self.with_mask_ref, use_early_fusion=self.use_early_fusion, num_frames=self.num_frames,
                backbone_channels=self.backbone_channels, use_backbone_feature=self.use_backbone_feature, use_iou_branch=self.use_iou_branch)
        self.detr.to(self.device)

        # Language Reference
        if self.with_lang_ref:
            self.lang_pool = cfg.MODEL.LANG_CONFIG.LANG_POOL
            lang_type = cfg.MODEL.LANG_CONFIG.MODEL_TYPE
            if lang_type == "roberta-base":
                self.tokenizer = RobertaTokenizerFast.from_pretrained('pretrained_models/roberta-base-uncased')
                self.text_encoder = RobertaModel.from_pretrained('pretrained_models/roberta-base-uncased')
            elif lang_type == "bert-base":
                self.tokenizer = BertTokenizerFast.from_pretrained('pretrained_models/bert-base-uncased')
                self.text_encoder = BertModel.from_pretrained('pretrained_models/bert-base-uncased')
            elif lang_type == "bert-large":
                self.tokenizer = BertTokenizerFast.from_pretrained('pretrained_models/bert-large-uncased')
                self.text_encoder = BertModel.from_pretrained('pretrained_models/bert-large-uncased')
            elif lang_type == "clip-base":
                self.tokenizer = CLIPTokenizerFast.from_pretrained('pretrained_models/clip-vit-base-patch32')
                self.text_encoder = CLIPTextModel.from_pretrained('pretrained_models/clip-vit-base-patch32')
            else:
                raise NotImplementedError("Language model not supported!")

            if cfg.MODEL.LANG_CONFIG.FREEZE_TEXT_ENCODER:
                for p in self.text_encoder.parameters():
                    p.requires_grad_(False)
            # resize the llm output channel to transformer d_model
            self.resizer = FeatureResizer(
                input_feat_size=cfg.MODEL.LANG_CONFIG.LANG_DIM,
                output_feat_size=hidden_dim,
                dropout=0.1,
                do_ln=True
            )
            self.context_len = cfg.MODEL.LANG_CONFIG.CONTEXT_LEN
        
        # Mask Reference
        if self.with_mask_ref:
            self.value_encoder = ValueEncoderSO()

        # --------------------------------------------------------------------------------
        # building criterion
        matcher = HungarianMatcher(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou)

        # build weight dict
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight}
        
        self.mask_aux_loss = cfg.MODEL.DDETRS.MASK_AUX_LOSS
        if self.mask_on:
            weight_dict["loss_mask"] = mask_weight
            weight_dict["loss_dice"] = dice_weight
            if self.mask_aux_loss:
                weight_dict["loss_mask_cls"] = mask_aux_weight
                weight_dict["loss_mask_iou"] = mask_aux_weight

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # losses = ["labels", "boxes", "cardinality"]
        # remove 'cardinality', otherwise the log loss will be very large
        losses = ["labels", "boxes"]
        if self.mask_on:
            if cfg.MODEL.BOXINST.ENABLED:   # objects365/vg pretrain
                losses += ["masks_boxinst"]
            else:
                losses += ["masks"]
        
        # Criterion
        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses, focal_alpha=focal_alpha, ota=self.ota,
                mask_out_stride=self.mask_stride,  mask_aux_loss=self.mask_aux_loss, cfg=cfg)
        self.criterion.to(self.device)

        self.to_agnostic = cfg.MODEL.DDETRS.TO_AGNOSTIC
        if self.to_agnostic:
            assert self.num_classes == 1, "Please set NUM_CLASSES=1 in the agnostic mode."

        # -------------------------------------------------------------------------
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.use_lsj = cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj"

        # BoxInst
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH

        # inference setting
        self.merge_on_cpu = cfg.MODEL.MERGE_ON_CPU
        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "inference", "refytvos") # for saving ref-youtube results
        self.save_refdavis_prefix = os.path.join(cfg.OUTPUT_DIR, "inference", "refdavis") # for saving ref-davis results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # judge task
        task = batched_inputs[0]["task"]

        # preprocess images
        if task in ["detection", "grounding", "fss"]:
            images = self.preprocess_image(batched_inputs)       # [b, c, h, w] 
        elif task in ["vos", "rvos"]:
            images = self.preprocess_clip_image(batched_inputs)  # [bxt, c, h, w]
        else:
            raise NotImplementedError(f"{task} is not supported.")
        # batched_inputs: 'height', 'width' original size
        # images.image_sizes: after aug, before padding
        # images.shape[-2:]: after aug and padding
        
        # training
        if self.training:
            # prepare targets
            if self.boxinst_enabled: 
                # obj365/vg pretrain
                images, targets = self.prepare_image_targets_boxinst(batched_inputs)
            else:  
                # image tasks
                if task in ["detection", "grounding", "fss"]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                # video tasks
                else:
                    gt_instances = []
                    for video in batched_inputs:
                        for frame in video["instances"]:
                            gt_instances.append(frame.to(self.device))
                targets = self.prepare_targets(gt_instances)

            if task == "detection":
                assert self.num_frames == 1
                output, loss_dict = self.detr.coco_forward(images, targets, self.criterion, train=True, task=task)
            elif task == "grounding":
                assert self.num_frames == 1
                captions = [x["expressions"] for x in batched_inputs] # list[str]
                lang_dict_features = self.forward_text(captions, device="cuda")
                output, loss_dict = self.detr.coco_forward(images, targets, self.criterion, train=True, 
                                                    lang_dict_features=lang_dict_features, task=task)
            elif task == "fss":
                assert self.num_frames == 1
                support_images = [x["support_images"][0] for x in batched_inputs]
                ref_images = self.preprocess_support_images(support_images)
                ref_gt_instances = [x["support_instances"][0].to(self.device) for x in batched_inputs]
                ref_targets = self.prepare_targets(ref_gt_instances)
                ref_masks = [target["masks"] for target in ref_targets]  # list[tensor], length of bs, [1, H, W], image size
                # forward to get the mask_dict_features
                srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(ref_images)  # multi-scale features, 8x -> 64x
                mask_dict_features = self.forward_template(ref_images, ref_masks, srcs_ref, masks_ref)  
                output, loss_dict = self.detr.coco_forward(images, targets, self.criterion, train=True,
                                                    mask_dict_features=mask_dict_features, task=task)
            elif task == "vos":
                assert self.num_frames == 2
                bz = len(images) // self.num_frames
                ref_ids = list(range(0, bz*self.num_frames, self.num_frames))  # first frame (gt)
                det_ids = list(range(1, bz*self.num_frames, self.num_frames))  # second frame
                # reference frame
                ref_images = ImageList.from_tensors([images[id] for id in ref_ids])
                ref_gt_instances = [gt_instances[id] for id in ref_ids]
                ref_targets = self.prepare_targets(ref_gt_instances)
                ref_masks = [target["masks"] for target in ref_targets]  # list[tensor], length of bs, [1, H, W], image size
                # forward to get the mask_dict_features
                srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(ref_images)  # multi-scale features, 8x -> 64x
                mask_dict_features = self.forward_template(ref_images, ref_masks, srcs_ref, masks_ref)
                # current frame
                det_images = ImageList.from_tensors([images[id] for id in det_ids])
                det_gt_instances = [gt_instances[id] for id in det_ids]
                det_targets = self.prepare_targets(det_gt_instances)  
                output, loss_dict = self.detr.coco_forward(det_images, det_targets, self.criterion, train=True,
                                                    mask_dict_features=mask_dict_features, task=task)
            elif task == "rvos":
                assert self.num_frames == 2
                dataset_name = batched_inputs[0]["dataset_name"]
                bz = len(images) // self.num_frames
                ref_ids = list(range(0, bz*self.num_frames, self.num_frames))  # first frame
                det_ids = list(range(1, bz*self.num_frames, self.num_frames))  # second frame
                det_images = ImageList.from_tensors([images[id] for id in det_ids])
                det_gt_instances = [gt_instances[id] for id in det_ids]
                det_targets = self.prepare_targets(det_gt_instances)  
                # language reference
                captions = [x["expressions"] for x in batched_inputs]  # list[str]
                lang_dict_features = self.forward_text(captions, device="cuda")
                # refcoco, only fuse with language
                if dataset_name == "video-refcoco":  
                    output, loss_dict = self.detr.coco_forward(det_images, det_targets, self.criterion, train=True, 
                                                    lang_dict_features=lang_dict_features, task=task)
                # rvos data
                else:  
                    if 0. < torch.rand(1) < 0.4:  # only fuse with language
                        output, loss_dict = self.detr.coco_forward(det_images, det_targets, self.criterion, train=True, 
                                                    lang_dict_features=lang_dict_features, task=task)
                    else:  # fuse with language and mask
                        ref_images = ImageList.from_tensors([images[id] for id in ref_ids])
                        ref_gt_instances = [gt_instances[id] for id in ref_ids]
                        ref_targets = self.prepare_targets(ref_gt_instances)
                        ref_masks = [target["masks"] for target in ref_targets]  # list[tensor], length of bs, [1, H, W], image size
                        # forward to get the mask_dict_features
                        srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(ref_images)  # multi-scale features, 8x -> 64x
                        mask_dict_features = self.forward_template(ref_images, ref_masks, srcs_ref, masks_ref)
                        output, loss_dict = self.detr.coco_forward(det_images, det_targets, self.criterion, train=True,
                                    lang_dict_features=lang_dict_features, mask_dict_features=mask_dict_features, task=task)

            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        # -----------------------------------------------------------------------------------------
        # inference
        else:
            # image task
            if task in ["detection", "grounding", "fss"]:
                if task == "detection":
                    output, loss_dict = self.detr.coco_inference(images, None, self.criterion, train=False, task=task)
                elif task == "grounding":
                    captions = [x["expressions"] for x in batched_inputs]  # list[str]
                    lang_dict_features = self.forward_text(captions, device="cuda")
                    output, loss_dict = self.detr.coco_inference(images, None, self.criterion, train=False,
                                                    lang_dict_features=lang_dict_features, task=task)
                elif task == "fss":
                    # batch_size = 1
                    support_images = batched_inputs[0]["support_images"]       # list[Tensor], length of nshot
                    ref_images = self.preprocess_support_images(support_images)
                    ref_gt_instances = batched_inputs[0]["support_instances"]  # list[Instance], length of nshot
                    ref_gt_instances = [x.to(self.device) for x in ref_gt_instances]
                    ref_targets = self.prepare_targets(ref_gt_instances)
                    ref_masks = [target["masks"] for target in ref_targets]  # list[tensor], length of bs, [1, H, W], image size
                    # forward to get the mask_dict_features
                    srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(ref_images)  # multi-scale features, 8x -> 64x
                    mask_dict_features = self.forward_template(ref_images, ref_masks, srcs_ref, masks_ref, aggregate=True)  
                    output, loss_dict = self.detr.coco_inference(images, None, self.criterion, train=False,
                                                        mask_dict_features=mask_dict_features, task=task) 

                # image task (coco, refcoco) post-process
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                # image inference: coco, refcoco
                results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, task=task)

                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    # here is the original image size
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})

                return processed_results
            # video task
            else:
                if task == "vos":
                    self.inference_vos_3f(batched_inputs)
                elif task == "rvos":
                    dataset_name = batched_inputs[0]["dataset_name"]
                    # aggregate_objects: whether aggregate objects in a single video
                    if dataset_name.startswith("refytvos"):
                        self.inference_rvos_vl(batched_inputs, images, aggregate_objects=False)
                    elif dataset_name.startswith("refdavis"):
                        self.inference_rvos(batched_inputs, images, aggregate_objects=True)
                return

    def prepare_targets(self, targets):
        # padding gt_masks to max size over a batch (This is important for training with propagation)
        if hasattr(targets[0], "gt_masks"):
            # mask size: (n_inst, hm, wm)
            gt_masks_list = [x.gt_masks if self.use_lsj else x.gt_masks.tensor for x in targets]
            max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
            stride = 32 # size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride

        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            if self.to_agnostic:
                gt_classes = torch.zeros_like(gt_classes)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                                    "boxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                                    "boxes_unorm": targets_per_image.gt_boxes.tensor})
        return new_targets


    def prepare_targets_test(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            # import pdb;pdb.set_trace()
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                # padding gt_masks to max size over a batch (This is important for training with img pairs)
                # mask size: (n_inst, hm, wm)
                gt_masks_list = [targets_per_image.gt_masks if self.use_lsj else targets_per_image.gt_masks.tensor]
                max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
                stride = 32 # size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
                max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                                    "boxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                                    "boxes_unorm": targets_per_image.gt_boxes.tensor})
        return new_targets

    # boxinst
    def prepare_image_targets_boxinst(self, batched_inputs, size_divisibility=32):
        original_images = [x["image"].to(self.device) for x in batched_inputs] # [tensor((3,H,W))] len=bs
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm)

        original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images] # [torch.ones(H, W),...] len=bs

        # mask out the bottom area where the COCO dataset probably has wrong annotations
        for i in range(len(original_image_masks)):
            im_h = batched_inputs[i]["height"]
            pixels_removed = int(
                self.bottom_pixels_removed *
                float(original_images[i].size(1)) / float(im_h)
            )
            if pixels_removed > 0:
                original_image_masks[i][-pixels_removed:, :] = 0

        original_images = ImageList.from_tensors(original_images, size_divisibility)
        original_image_masks = ImageList.from_tensors(
            original_image_masks, size_divisibility, pad_value=0.0
        ) # (bs, H, W) image=1, padding=0
        self.add_bitmasks_from_boxes(
            gt_instances, original_images.tensor, original_image_masks.tensor,
            original_images.tensor.size(-2), original_images.tensor.size(-1)
        )
        
        new_targets = self.prepare_targets_boxinst(gt_instances)
        
        return images_norm, new_targets

    def prepare_targets_boxinst(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if self.use_lsj:
                raise NotImplementedError
            else:
                gt_masks = targets_per_image.gt_bitmasks_full
            if self.use_amp:
                gt_masks = gt_masks.half()
            image_color_similarity = targets_per_image.image_color_similarity
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, 
                                "image_color_similarity": image_color_similarity})
        return new_targets


    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        stride = self.mask_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]] # RGB-format original images (with padding) (bs, 3, H//4, W//4)
        image_masks = image_masks[:, start::stride, start::stride] # (bs, H//4, W//4)

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy()) # (H, W, 3)
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None] # (1, 3, H//4, W//4)
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation
            ) # (1, 8, H//4, W//4)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0) # (N, H, W)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst)) # (N, 8, H//4, W//4)
            ], dim=0)


    # inference based on three frames: the 1st frame, frame T-1, frame T   
    def inference_vos_3f(self, batched_inputs, task="vos"):
        # inference batch size is 1
        init_file_name = batched_inputs[0]["file_names"][0]
        dataset_name = batched_inputs[0]["dataset_name"]
        # dataset_name = init_file_name.split("/")[1]
        if dataset_name == "ytbvos18" or dataset_name == "ytbvos19":
            palette_img = "datasets/ytbvos18/valid/Annotations/0a49f5265b/00000.png"
            score_thres = 0.4
        elif dataset_name == "davis17":
            palette_img = "datasets/davis17/DAVIS/Annotations/480p/bear/00000.png"
            score_thres = 0.3
        elif dataset_name == "vos-lvos":
            palette_img = "datasets/lvos/valid/Annotations/0tCWPOrc/00000001.png"
            score_thres = 0.5
        elif dataset_name == "mose":
            palette_img = "datasets/mose/valid/Annotations/00c28e4b/00000.png"
            score_thres = 0.4
        else:
            raise ValueError
        palette = Image.open(palette_img).getpalette()
        # youtube-vos and davis follow the same inference process
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        video_len = len(batched_inputs[0]["image"])
        gt_instances = batched_inputs[0]["instances"]
        gt_instances = [x.to(self.device) for x in gt_instances]  
        gt_targets = self.prepare_targets_test(gt_instances)  # mask pad to size_divisibility=32
        assert len(gt_targets) == video_len
        file_names = batched_inputs[0]["file_names"]
        vid_name = init_file_name.split("/")[-2]
        save_dir = os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # begin tracking
        ref_dict_init = {}  # the gt frame
        ref_dict_prev = {}  # the previous frame
        mask_file_names = [x.split("/")[-1].replace(".jpg", ".png") for x in file_names]
        for frame_idx in range(video_len):
            # frame_idx indicate current frame
            clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]   # one frame
            images = self.preprocess_clip_image(clip_inputs)
            # reference image features, only forward once
            srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(images)  # multi-scale features, 8x -> 64x

            # step 1: get the reference mask
            cur_gt_instances = gt_instances[frame_idx]
            cur_new_obj_ids = []
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, initialize templates for them
                ref_masks = gt_targets[frame_idx]["masks"]          # tensor size of [N, H, W]
                cur_obj_ids = cur_gt_instances.ori_id
                num_new_obj = len(cur_obj_ids)
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    cur_new_obj_ids.append(cur_obj_id)
                    cur_ref_masks = [ref_masks[obj_idx:obj_idx+1]]  # list, [1, H, W]
                    assert cur_obj_id not in ref_dict_init
                    init_mask_dict_features = self.forward_template(images, cur_ref_masks, srcs_ref, masks_ref)
                    # when the object appears for the first time, init and prev use the same ref_feats
                    ref_dict_init[cur_obj_id] = init_mask_dict_features
                    ref_dict_prev[cur_obj_id] = init_mask_dict_features

            # we start tracking a new video from the first given GT frame, e.g. 00f88c4f0a
            if len(cur_gt_instances) == 0 and len(list(ref_dict_init.keys())) == 0: 
                continue
            
            # step 2: predict the mask in current frame
            # track
            mask_dict = {} # store mask results of different obj_id on the current frame
            for obj_id in ref_dict_init.keys():
                ref_feats_init = copy.deepcopy(ref_dict_init[obj_id])   # important
                ref_feats_prev = copy.deepcopy(ref_dict_prev[obj_id])   # important
                # concat mask_dict_features
                cur_mask_dict_features = concat_mask_dict_features(ref_feats_init, ref_feats_prev)
                output, _ = self.detr.coco_inference(images, None, None, train=False,
                                        mask_dict_features=cur_mask_dict_features, task=task)
                cls_pred = output["pred_logits"] # [1, Q, 1]
                box_pred = output["pred_boxes"] # [1, Q, 4]
                mask_pred = output["pred_masks"] if self.mask_on else [None] * len(batched_inputs)
                output_h, output_w = mask_pred.shape[-2:]
                num_inst = 1  # vos only has one object
                # loop over  each image, online fashion for 1 time
                for _, (logits, output_masks, output_boxes, image_size) in enumerate(zip(
                    cls_pred, mask_pred, box_pred, images.image_sizes
                )):
                    prob = logits.sigmoid()  # [300, 1]
                    topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                    indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                    # [0, 1] -> real coordinates
                    output_boxes[:, 0::2] *= width
                    output_boxes[:, 1::2] *= height
                    # get the final results
                    # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                    if topk_values > score_thres:
                        track_masks = output_masks[indices] # [N_obj, 1 ,H, W], N_obj is 1
                        track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride_vos, output_w*self.mask_stride_vos) ,mode="bilinear", align_corners=False).sigmoid()
                        track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                        mask_dict[obj_id] = track_masks[0, 0] # [H, W]
                    else:
                        track_masks = torch.zeros((height, width), device=self.device)
                        mask_dict[obj_id] = track_masks
                
            # deal with objects appearing in the current frame
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, replace predicted mask results with gts
                cur_obj_ids = cur_gt_instances.ori_id
                num_new_obj = len(cur_obj_ids)
                gt_masks = gt_targets[frame_idx]["masks"][:, :image_size[0], :image_size[1]] # here image size is from the images.image_sizes above
                gt_masks = F.interpolate(gt_masks[None].float(), size=(height, width), mode="bilinear", align_corners=False)[0]
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    mask_dict[cur_obj_id] = gt_masks[obj_idx]

            # step 3: aggregates masks for all objects in current frame
            # post-process (soft-aggregation)
            cur_obj_ids = sorted(list(ref_dict_init.keys()))
            cur_obj_ids_int = [int(x) for x in cur_obj_ids] # 1, 2, 3...
            if len(cur_obj_ids_int) != 0:
                mask_merge = torch.zeros((height, width, max(cur_obj_ids_int)+1), device=self.device) # [H, W, N+1]
            else:
                mask_merge = torch.zeros((height, width, 1), device=self.device)
            tmp_list = []
            for cur_id in cur_obj_ids:
                mask_merge[:, :, int(cur_id)] = mask_dict[cur_id]
                tmp_list.append(mask_dict[cur_id])
            if len(tmp_list) != 0:
                back_prob = torch.prod(1 - torch.stack(tmp_list, dim=-1), dim=-1, keepdim=False)
                mask_merge[:, :, 0] = back_prob
            mask_merge = torch.argmax(mask_merge, dim=-1)
            mask_merge_final = mask_merge.cpu().numpy().astype(np.uint8) # [H, W]
            mask_merge_final = Image.fromarray(mask_merge_final).convert('P')
            mask_merge_final.putpalette(palette)
            save_img_dir = os.path.join(save_dir, vid_name)
            os.makedirs(save_img_dir, exist_ok=True)
            mask_merge_final.save(os.path.join(save_img_dir, mask_file_names[frame_idx]))

            # step 4: update ref_dict_prev
            for cur_id in cur_obj_ids:
                if cur_id in cur_new_obj_ids:
                    continue
                cur_mask = (mask_merge == int(cur_id))
                # cur_mask = mask_dict[cur_id] > 0.5 # [H, W], binary mask
                # the cur_mask is all 0, which means the object is not visiable 
                # in current frame, do not update the prev dict
                if not (cur_mask > 0).any():
                    continue
                try:
                    # from origin image size to image size
                    samples = nested_tensor_from_tensor_list(images, size_divisibility=32)
                    pad_h, pad_w = samples.tensors.shape[-2:]
                    cur_mask_tensor = cur_mask[None] # [1, H, W]
                    cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                    cur_mask_tensor_final = torch.zeros((1, pad_h, pad_w), device=self.device)
                    cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                    cur_ref_masks = [cur_mask_tensor_final]
                    cur_mask_dict_features = self.forward_template(images, cur_ref_masks, srcs_ref, masks_ref)
                    ref_dict_prev[cur_id] = cur_mask_dict_features
                except:
                    # this could happen when cur_mask == 0
                    print(f"warning: {vid_name}_{frame_idx} failed.")
                    continue
        # print("%s done."%vid_name)

    def inference_rvos_vl(self, batched_inputs, images, aggregate_objects=False, task="rvos"):
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        video_length = len(batched_inputs[0]['file_names'])
        # during inference, batch size always 1 per gpu
        # for refytvos, each expression is saved in a single folder
        if not aggregate_objects:
            dataset_name = batched_inputs[0]["dataset_name"]
            assert dataset_name == "refytvos"
            score_thres = 0.3 

            # images: [video_length, c, h, w]
            self.detr.num_frames = 1
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            lang_dict_features = self.forward_text(captions, device="cuda")

            # store the first and prev masks
            ref_init_dict = {}
            ref_prev_dict = {}

            # get the final_masks
            final_masks = []
            # inference for each frame
            for frame_idx in range(video_length):
                # step 1: first frame, predict mask
                if not ref_init_dict:  # the first frame, fuse L to generate mask
                    clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                    clip_images = self.preprocess_clip_image(clip_inputs)
                    outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                                        lang_dict_features=lang_dict_features, task=task)
                
                    # inference each frame independently
                    video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                    video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                    video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                    output_h, output_w = video_output_masks.shape[-2:]
                    num_inst = 1  # rvos only has one object
                    # loop over each image
                    for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                        video_logits, video_output_masks, video_output_boxes, images.image_sizes
                    )):
                        prob = logits.sigmoid()   # [300, 1]
                        topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                        indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                        # [0, 1] -> real coordinates
                        output_boxes[:, 0::2] *= width
                        output_boxes[:, 1::2] *= height
                        # get the final results
                        # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                        if topk_values > score_thres:
                            track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                            track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid().to()
                            track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                            track_masks = track_masks[0, 0]    
                            final_masks.append((track_masks > 0.5).cpu().numpy())
                        else:
                            track_masks = torch.zeros((height, width), device=self.device)
                            final_masks.append((track_masks > 0.5).cpu().numpy())
                    del outputs

                    # store the masks to ref_init_dict and ref_prev_dict
                    cur_mask = track_masks > 0.5 # binary mask
                    if (cur_mask > 0).any():
                        # from origin image size to image size
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=32)
                        pad_h, pad_w = samples.tensors.shape[-2:]
                        cur_mask_tensor = cur_mask[None] # [1, H, W]
                        cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                        cur_mask_tensor_final = torch.zeros((1, pad_h, pad_w), device=self.device)
                        cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                        cur_ref_masks = [cur_mask_tensor_final]
                        srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(clip_images)
                        cur_mask_dict_features = self.forward_template(clip_images, cur_ref_masks, srcs_ref, masks_ref)
                        ref_init_dict.update(cur_mask_dict_features)
                        ref_prev_dict.update(cur_mask_dict_features)
                # step 2: following frames, start tracking with langugage and predicted mask
                else:  
                    clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                    clip_images = self.preprocess_clip_image(clip_inputs)

                    ref_feats_init = copy.deepcopy(ref_init_dict)
                    ref_feats_prev = copy.deepcopy(ref_prev_dict)
                    # concat mask_dict_features
                    cur_mask_dict_features = concat_mask_dict_features(ref_feats_init, ref_feats_prev)
                    # fuse V + L
                    outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                        lang_dict_features=lang_dict_features, mask_dict_features=cur_mask_dict_features, task=task)

                    # inference each frame independently
                    video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                    video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                    video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                    output_h, output_w = video_output_masks.shape[-2:]
                    num_inst = 1  # rvos only has one object
                    # loop over each image
                    for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                        video_logits, video_output_masks, video_output_boxes, images.image_sizes
                    )):
                        prob = logits.sigmoid()   # [300, 1]
                        topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                        indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                        # [0, 1] -> real coordinates
                        output_boxes[:, 0::2] *= width
                        output_boxes[:, 1::2] *= height
                        # get the final results
                        # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                        if topk_values > score_thres:
                            track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                            track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid().to()
                            track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                            track_masks = track_masks[0, 0]    
                            final_masks.append((track_masks > 0.5).cpu().numpy())
                        else:
                            track_masks = torch.zeros((height, width), device=self.device)
                            final_masks.append((track_masks > 0.5).cpu().numpy())
                    del outputs

                    # step 3: update ref_prev_dict
                    cur_mask = track_masks > 0.5 # binary mask
                    if (cur_mask > 0).any(): 
                        # from origin image size to image size
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=32)
                        pad_h, pad_w = samples.tensors.shape[-2:]
                        cur_mask_tensor = cur_mask[None] # [1, H, W]
                        cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                        cur_mask_tensor_final = torch.zeros((1, pad_h, pad_w), device=self.device)
                        cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                        cur_ref_masks = [cur_mask_tensor_final]
                        srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(clip_images)
                        cur_mask_dict_features = self.forward_template(clip_images, cur_ref_masks, srcs_ref, masks_ref)
                        ref_prev_dict = cur_mask_dict_features

            video_name, exp_id = batched_inputs[0]["video"], batched_inputs[0]["exp_id"]
            # save binary image
            save_path = os.path.join(self.save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for frame_idx in range(video_length):
                frame_name = batched_inputs[0]["file_names"][frame_idx].split("/")[-1].replace(".jpg", "")
                mask = final_masks[frame_idx].astype(np.float32) 
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)
            return 
        # for ref-davis, all objects should be aggregated in one video
        else:  
            dataset_name = batched_inputs[0]["dataset_name"]
            assert "refdavis" in dataset_name
            palette_img = "datasets/ref-davis/DAVIS/Annotations_unsupervised/480p/bear/00000.png"
            palette = Image.open(palette_img).getpalette()

            video_name = batched_inputs[0]["video"]
            file_names = batched_inputs[0]["file_names"]
            mask_file_names = [x.split("/")[-1].replace(".jpg", ".png") for x in file_names]

            # inference batch size = 1
            expressions = batched_inputs[0]["expressions"] # ["exp1", "exp2", ...]
            num_expressions = len(expressions) # i.e. num_object

            all_final_masks = [] # list[list[np.array]], first list: all_objects, second list: all_frames

            # 1. for each object
            for exp_id in range(num_expressions):
                # get final masks of a expression, 
                # list[np.array], length is video_len, array size of [H, W], indicates the mask scores
                final_masks = []
                # images: [video_length, c, h, w]
                self.detr.num_frames = 1
                # captions: list[str]
                captions = [expressions[exp_id]]
                lang_dict_features = self.forward_text(captions, device="cuda")

                # store the init and prev masks
                ref_init_dict = {}
                ref_prev_dict = {}

                # 2. for each frame
                for frame_idx in range(video_length):
                    # step 1: first frame, predict mask
                    if not ref_init_dict:
                        clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                        clip_images = self.preprocess_clip_image(clip_inputs)
                        outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                                            lang_dict_features=lang_dict_features, task=task)
                        # inference each frame independently
                        video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                        video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                        video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                        output_h, output_w = video_output_masks.shape[-2:]
                        num_inst = 1  # rvos only has one object
                        # loop over each image, online fashion only for 1 time
                        for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                            video_logits, video_output_masks, video_output_boxes, images.image_sizes
                        )):
                            prob = logits.sigmoid()   # [300, 1]
                            topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                            indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                            # [0, 1] -> real coordinates
                            output_boxes[:, 0::2] *= width
                            output_boxes[:, 1::2] *= height
                            # get the final results
                            # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                            track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                            track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid().to()
                            track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                            track_masks = track_masks[0, 0]  # (H, W), mask scores
                            final_masks.append(track_masks)
                        del outputs

                        # store the masks to ref_init_dict and ref_prev_dict
                        cur_mask = track_masks > 0.5
                        # davis17, all objects appear in the first frame
                        # from origin image size to image size
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=32)
                        pad_h, pad_w = samples.tensors.shape[-2:]
                        cur_mask_tensor = cur_mask[None] # [1, H, W]
                        cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                        cur_mask_tensor_final = torch.zeros((1, pad_h, pad_w), device=self.device)
                        cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                        cur_ref_masks = [cur_mask_tensor_final]
                        srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(clip_images)
                        cur_mask_dict_features = self.forward_template(clip_images, cur_ref_masks, srcs_ref, masks_ref)
                        ref_init_dict.update(cur_mask_dict_features)
                        ref_prev_dict.update(cur_mask_dict_features)
                    # step 2: following frames, start tracking with language and predicted mask
                    else:
                        clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                        clip_images = self.preprocess_clip_image(clip_inputs)

                        ref_feats_init = copy.deepcopy(ref_init_dict)
                        ref_feats_prev = copy.deepcopy(ref_prev_dict)
                        # concat mask_dict_features
                        cur_mask_dict_features = concat_mask_dict_features(ref_feats_init, ref_feats_prev)
                        # fuse V + L
                        outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                            lang_dict_features=lang_dict_features, mask_dict_features=cur_mask_dict_features, task=task)

                        # inference each frame independently
                        video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                        video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                        video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                        output_h, output_w = video_output_masks.shape[-2:]
                        num_inst = 1  # rvos only has one object
                        # loop over each image
                        for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                            video_logits, video_output_masks, video_output_boxes, images.image_sizes
                        )):
                            prob = logits.sigmoid()   # [300, 1]
                            topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                            indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                            # [0, 1] -> real coordinates
                            output_boxes[:, 0::2] *= width
                            output_boxes[:, 1::2] *= height
                            # get the final results
                            # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                            track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                            track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid().to()
                            track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                            track_masks = track_masks[0, 0]    # (H, W), mask score
                            final_masks.append(track_masks)
                        del outputs

                        # step 3: update ref_prev_dict
                        cur_mask = track_masks > 0.5 # binary mask
                        if (cur_mask > 0).any(): 
                            # from origin image size to image size
                            samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=32)
                            pad_h, pad_w = samples.tensors.shape[-2:]
                            cur_mask_tensor = cur_mask[None] # [1, H, W]
                            cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                            cur_mask_tensor_final = torch.zeros((1, pad_h, pad_w), device=self.device)
                            cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                            cur_ref_masks = [cur_mask_tensor_final]
                            srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(clip_images)
                            cur_mask_dict_features = self.forward_template(clip_images, cur_ref_masks, srcs_ref, masks_ref)
                            ref_prev_dict = cur_mask_dict_features
                # store the current object all mask scores
                all_final_masks.append(torch.stack(final_masks, dim=0))
                
            # post-process, mask merge by soft-aggregation
            # all_final_masks:
            #   list[list[tensor]], first list is all_objects, second list is all_frames
            #   size of [H, W]
            all_final_masks = torch.stack(all_final_masks, dim=0) # [N, T, H, W]
            cur_obj_ids_int = [int(x) for x in range(num_expressions)]  # num_expressions = num_objects, 0, 1, 2
            for frame_idx in range(video_length):
                cur_masks = all_final_masks[:, frame_idx, :, :]   # [N_obj, H, W]
                # NOTE: mask_merge has additional background channel
                mask_merge = torch.zeros((height, width, len(cur_obj_ids_int)+1), device=self.device) # [H, W, N_obj+1]
                tmp_list = []
                for cur_id in cur_obj_ids_int:
                    mask_merge[:, :, cur_id+1] = cur_masks[cur_id]
                    tmp_list.append(cur_masks[cur_id])
                if len(tmp_list) != 0: # calculate the background prob
                    back_prob = torch.prod(1 - torch.stack(tmp_list, dim=-1), dim=-1, keepdim=False)
                    mask_merge[:, :, 0] = back_prob
                mask_merge = torch.argmax(mask_merge, dim=-1)
                mask_merge_final = mask_merge.cpu().numpy().astype(np.uint8) # (H, W)
                mask_merge_final = Image.fromarray(mask_merge_final).convert('P')
                mask_merge_final.putpalette(palette)
                save_img_dir = os.path.join(self.save_refdavis_prefix, dataset_name, video_name)
                os.makedirs(save_img_dir, exist_ok=True)
                mask_merge_final.save(os.path.join(save_img_dir, mask_file_names[frame_idx]))
            return


    def inference_rvos(self, batched_inputs, images, aggregate_objects=False, task="rvos"):
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        video_length = len(batched_inputs[0]['file_names'])
        # during inference, batch size always 1 per gpu
        # for refytvos, each expression is saved in a single folder
        if not aggregate_objects: 
            dataset_name = batched_inputs[0]["dataset_name"]
            assert dataset_name == "refytvos"
            score_thres = 0.3

            # images: [video_length, c, h, w]
            self.detr.num_frames = 1
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            lang_dict_features = self.forward_text(captions, device="cuda")

            # get the final_masks
            final_masks = []
            # inference for each frame
            for frame_idx in range(video_length):
                clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                clip_images = self.preprocess_clip_image(clip_inputs)
                outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                                lang_dict_features=lang_dict_features, task=task)
            
                # inference each frame independently
                video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                output_h, output_w = video_output_masks.shape[-2:]
                num_inst = 1  # rvos only has one object
                # loop over each image
                for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                    video_logits, video_output_masks, video_output_boxes, images.image_sizes
                )):
                    prob = logits.sigmoid()   # [300, 1]
                    topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                    indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                    # [0, 1] -> real coordinates
                    output_boxes[:, 0::2] *= width
                    output_boxes[:, 1::2] *= height
                    # get the final results
                    # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                    if topk_values > score_thres:
                        track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                        track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid()
                        track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                        track_masks = track_masks[0, 0] # [H, W], mask scores
                        final_masks.append((track_masks > 0.5).cpu().numpy())
                    else: 
                        track_masks = torch.zeros((height, width), device=self.device)
                        final_masks.append((track_masks > 0.5).cpu().numpy())
                del outputs

            video_name, exp_id = batched_inputs[0]["video"], batched_inputs[0]["exp_id"]
            # save binary image
            save_path = os.path.join(self.save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for frame_idx in range(video_length):
                frame_name = batched_inputs[0]["file_names"][frame_idx].split("/")[-1].replace(".jpg", "")
                mask = final_masks[frame_idx].astype(np.float32) 
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)
            return 
        # for ref-davis, all objects are aggregated in one video
        else:  
            dataset_name = batched_inputs[0]["dataset_name"]
            assert "refdavis" in dataset_name
            palette_img = "datasets/ref-davis/DAVIS/Annotations_unsupervised/480p/bear/00000.png"
            palette = Image.open(palette_img).getpalette()

            video_name = batched_inputs[0]["video"]
            file_names = batched_inputs[0]["file_names"]
            mask_file_names = [x.split("/")[-1].replace(".jpg", ".png") for x in file_names]

            # inference batch size = 1
            expressions = batched_inputs[0]["expressions"] # ["exp1", "exp2", ...]
            num_expressions = len(expressions) # i.e. num_object

            all_final_masks = [] # list[list[np.array]], first list: all_objects, second list: all_frames

            # 1. for each object
            for exp_id in range(num_expressions):
                # get final masks of a expression, 
                # list[np.array], length is video_len, array size of [H, W], indicates the mask scores
                final_masks = []
                # images: [video_length, c, h, w]
                self.detr.num_frames = 1
                # captions: list[str]
                captions = [expressions[exp_id]]
                lang_dict_features = self.forward_text(captions, device="cuda")
                # 2. for each frame
                for frame_idx in range(video_length):
                    clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                    clip_images = self.preprocess_clip_image(clip_inputs)
                    outputs, _ = self.detr.coco_inference(clip_images, None, self.criterion, train=False, 
                                                lang_dict_features=lang_dict_features, task=task)
                
                    # inference each frame independently
                    video_logits = outputs["pred_logits"]       # [1, 300, 1], 
                    video_output_boxes = outputs['pred_boxes']  # [1, 300, 4], cxcywh
                    video_output_masks = outputs["pred_masks"]  # [1, 300, 1, H/4, W/4]
                    output_h, output_w = video_output_masks.shape[-2:]
                    num_inst = 1  # rvos only has one object
                    # loop over each image, online fashion only for 1 time
                    for _, (logits, output_mask, output_boxes, image_size) in enumerate(zip(
                        video_logits, video_output_masks, video_output_boxes, images.image_sizes
                    )):
                        prob = logits.sigmoid()   # [300, 1]
                        topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)  # [1,]
                        indices = torch.div(topk_indexes, logits.shape[1], rounding_mode='floor')
                        # [0, 1] -> real coordinates
                        output_boxes[:, 0::2] *= width
                        output_boxes[:, 1::2] *= height
                        # get the final results
                        # track_bboxes = box_cxcywh_to_xyxy(output_boxes[indices])
                        track_masks = output_mask[indices] # (N_obj, 1, H, W), N_obj is 1
                        track_masks = F.interpolate(track_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid()
                        track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # (1, 1, H, W), resize to original size
                        track_masks = track_masks[0, 0]  # (H, W), mask scores
                        final_masks.append(track_masks)
                    del outputs
                # store the current object all mask scores
                all_final_masks.append(torch.stack(final_masks, dim=0))
                
            # post-process, mask merge by soft-aggregation
            # all_final_masks:
            #   list[list[tensor]], first list is all_objects, second list is all_frames
            #   size of [H, W]
            all_final_masks = torch.stack(all_final_masks, dim=0) # [N, T, H, W]
            cur_obj_ids_int = [int(x) for x in range(num_expressions)]  # num_expressions = num_objects, 0, 1, 2
            for frame_idx in range(video_length):
                cur_masks = all_final_masks[:, frame_idx, :, :]   # [N_obj, H, W]
                # NOTE: mask_merge has additional background channel
                mask_merge = torch.zeros((height, width, len(cur_obj_ids_int)+1), device=self.device) # [H, W, N_obj+1]
                tmp_list = []
                for cur_id in cur_obj_ids_int:
                    mask_merge[:, :, cur_id+1] = cur_masks[cur_id]
                    tmp_list.append(cur_masks[cur_id])
                if len(tmp_list) != 0: # calculate the background prob
                    back_prob = torch.prod(1 - torch.stack(tmp_list, dim=-1), dim=-1, keepdim=False)
                    mask_merge[:, :, 0] = back_prob
                mask_merge = torch.argmax(mask_merge, dim=-1)
                mask_merge_final = mask_merge.cpu().numpy().astype(np.uint8) # (H, W)
                mask_merge_final = Image.fromarray(mask_merge_final).convert('P')
                mask_merge_final.putpalette(palette)
                save_img_dir = os.path.join(self.save_refdavis_prefix, dataset_name, video_name)
                os.makedirs(save_img_dir, exist_ok=True)
                mask_merge_final.save(os.path.join(save_img_dir, mask_file_names[frame_idx]))
            return
            
        
    # inference for image tasks (coco, refcoco) 
    def inference(self, box_cls, box_pred, mask_pred, image_sizes, binary_mask=True, task="detection"):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size) in enumerate(zip(
            box_cls, box_pred, image_sizes
        )):
            if self.ota:
                # NMS
                prob = logits_per_image.sigmoid()       # [N, K]
                nms_scores, idxs = torch.max(prob,1)    # [N,], [N]
                boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                # import pdb;pdb.set_trace()
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,self.test_nms)  
                prob = prob[keep_indices]               # [Nl, K]
                box_pred_per_image = box_pred_per_image[keep_indices]
                if mask_pred is not None:
                    mask_pred_i = mask_pred[i][keep_indices]

                topk = min(sum(prob.flatten(0).shape), self.test_topk)
                if task != "detection":
                    topk = 1
                topk_values, topk_indexes = torch.topk(prob.view(-1), topk, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]
                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred is not None:
                    mask_pred_i = mask_pred_i[topk_boxes]
            else:
                prob = logits_per_image.sigmoid()
                topk = min(sum(prob.flatten(0).shape), self.test_topk)
                if task != "detection":
                    topk = 1
                topk_values, topk_indexes = torch.topk(prob.view(-1), topk, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]

                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred is not None:
                    mask_pred_i = mask_pred[i][topk_boxes]
            
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            #TODO batchsize>1 时关于size和padding需要修改
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            # import pdb;pdb.set_trace()
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                if binary_mask:
                    mask = mask.sigmoid() > self.mask_thres
                else:
                    mask = mask.sigmoid()
                # import pdb;pdb.set_trace()
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
                
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_support_images(self, support_images):
        images = [self.normalizer(image.to(self.device)) for image in support_images]
        images = ImageList.from_tensors(images)
        return images

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        if self.use_lsj and self.training:
            image_sizes = [x["instances"].image_size for x in batched_inputs]
            input_masks = [x["padding_mask"].to(self.device) for x in batched_inputs]
            H, W = images[0].size()[-2:]
            images_new = torch.zeros((len(images), 3, H, W), device=self.device)
            for i in range(len(images)):
                h, w = image_sizes[i]
                images_new[i, :, :h, :w] = images[i][:, :h, :w]
            outputs = NestedTensor(images_new, torch.stack(input_masks, dim=0))
            outputs.image_sizes = image_sizes
            return outputs
        else:
            images = ImageList.from_tensors(images)
            return images

    def preprocess_clip_image(self, batched_inputs, clip_idx=None):
        """
        Normalize, pad and batch the input images.
        """
        if clip_idx is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
        else:
            images = []
            for video in batched_inputs:
                for idx in clip_idx:
                    images.append(self.normalizer(video["image"][idx].to(self.device)))
            images = ImageList.from_tensors(images)
        return images

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(
                captions, padding="max_length", truncation=True, max_length=self.context_len, return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length], padding positions are 1

            text_features = encoded_text.last_hidden_state 
            text_features = self.resizer(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            if self.lang_pool:
                text_word_features, text_word_masks = text_features.decompose()
                embedded = text_word_features * text_word_masks.ne(1).unsqueeze(-1) # [batch_size, length, C]
                text_sentence_features = embedded.sum(1) / (text_word_masks.ne(1).sum(-1).unsqueeze(-1).float()) # [batch_size, C]
            else:
                text_sentence_features = encoded_text.pooler_output  
                text_sentence_features = self.resizer(text_sentence_features)  

            lang_dict_features = {}
            lang_dict_features["refs"] = text_word_features       # [bs, seq, c]
            lang_dict_features["ref_values"] = text_word_features # [bs, seq, c]
            lang_dict_features["masks"] = text_word_masks         # [bs, seq], bool, padding locations are 1
            lang_dict_features["ref_embeds"] = text_sentence_features # [bs, c]
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return lang_dict_features


    def forward_ref_backbone(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)

        features, pos = self.detr.detr.backbone(samples) # 8x, 16, 32x

        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, (feat, pos_l) in enumerate(zip(features[-self.num_transformer_features:], pos[-self.num_transformer_features:])):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.detr.num_feature_levels > len(features[-self.num_transformer_features:]):
            _len_srcs = len(features[-self.num_transformer_features:])
            for l in range(_len_srcs, self.detr.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        return srcs, masks, poses, spatial_shapes


    def forward_template(self, samples, masks, srcs_ref, masks_ref, aggregate=False):
        """
        Args:
            samples (NestedTensor)
                samples.tensors: [B, 3, H, W]
                samples.mask:    [B, H, W]
            masks: either gt or prediction, length is B
                masks (list[tensor]): [1, H, W]
            srcs_ref, masks_ref are the references frame features
            srcs_ref (list[tensor]):  [B, 3, Hi, Wi]
            masks_ref (list[tensor]): [B, Hi, Wi]
            aggregate (bool): ONLY used for fss few-shot inference, aggregate multiple support images
                # bs= 1
                "refs" (list[FloatTensor]): list of [1, \sum(HiWi), C]
                "ref_values" (list[FloatTensor]): list of [1, \sum(HiWi), C]
                "masks" (list[BoolTensor]): list of [1, \sum(HiWi)]
                "ref_emeds" (list[FloatTensor]): list of [1, C]       
        Returns:
            mask_dict_features (dict)
                "refs" (list[FloatTensor]): list of [B, HiWi, C]
                "ref_values" (list[FloatTensor]): list of [B, HiWi, C]
                "masks" (list[BoolTensor]): list of [B, HiWi]
                "ref_emeds" (list[FloatTensor]): list of [B, C]
        """
        # image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        # samples.tensors: [B, 3, H, W]
        # samples.mask:    [B, H, W], padding locations are 1
        
        # ref_tensor_list, ref_mask_list = [], []
        gt_mask_list = []
        for i in range(len(samples.tensors)):   # for B times
            # get mask in image size, [1, 1, H, W]
            gt_mask = masks[i][None]
            gt_mask_list.append(gt_mask)
        gt_masks = torch.cat(gt_mask_list, dim=0)       # [B, 1, H, W]
    
        # forward to get the reference frame multi-scale values, play as V
        values_ref = self.value_encoder(samples.tensors, gt_masks, srcs_ref)

        refs = []
        masks = []
        ref_values = [] 
        ref_embeds = []
        for n_l in range(self.detr.detr.num_feature_levels): # from 8x -> 64x
            # src: [B, C，Hi, Wi] -> [B, HiWi, C], mask: [B, Hi, Wi] -> [B, HiWi]
            ref_feat_l = srcs_ref[n_l].flatten(-2).permute(0, 2, 1)
            ref_mask_l = masks_ref[n_l].flatten(-2)
            ref_value_l = values_ref[n_l].flatten(-2).permute(0, 2, 1)
            embedded = ref_feat_l * ref_mask_l.ne(1).unsqueeze(-1)
            ref_embed_l = embedded.sum(1) / (ref_mask_l.ne(1).sum(-1).unsqueeze(-1).float()) # [B, C]
            if aggregate:
                ref_feat_l = ref_feat_l.flatten(0, 1).unsqueeze(0)
                ref_mask_l = ref_mask_l.flatten(0, 1).unsqueeze(0)
                ref_value_l = ref_value_l.flatten(0, 1).unsqueeze(0)
                ref_embed_l = ref_embed_l.mean(0).unsqueeze(0)
            refs.append(ref_feat_l)
            masks.append(ref_mask_l)
            ref_values.append(ref_value_l)
            ref_embeds.append(ref_embed_l)  # [B, C]            

        mask_dict_features = {}
        mask_dict_features["refs"] = refs              # list[FloatTensor]
        mask_dict_features["ref_values"] = ref_values  # list[FloatTensor]
        mask_dict_features["masks"] = masks            # list[BoolTensor]
        mask_dict_features["ref_embeds"] = ref_embeds  # list[FloatTensor]
        return mask_dict_features

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def concat_mask_dict_features(ref_feats_dict1, ref_feats_dict2):
    # ref_feat_dict (dict)
    #   "refs", "ref_values", "masks", "ref_embeds"
    assert isinstance(ref_feats_dict1, dict) and isinstance(ref_feats_dict2, dict)
    level = len(ref_feats_dict1["refs"])
    refs, ref_values, masks, ref_embeds = [], [], [], []
    for n_l in range(level):
        ref = torch.cat([ref_feats_dict1["refs"][n_l], ref_feats_dict2["refs"][n_l]], dim=1)  # [B, HiWi+HiWi, C]
        ref_value = torch.cat([ref_feats_dict1["ref_values"][n_l], ref_feats_dict2["ref_values"][n_l]], dim=1)  # [B, HiWi+HiWi, C]
        mask = torch.cat([ref_feats_dict1["masks"][n_l], ref_feats_dict2["masks"][n_l]], dim=1)  # [B, HiWi+HiWi]
        embedded = torch.cat([ref_feats_dict1["ref_embeds"][n_l].unsqueeze(1), ref_feats_dict2["ref_embeds"][n_l].unsqueeze(1)], dim=1)
        ref_embed = embedded.mean(1)  # [B, C]
        refs.append(ref)
        ref_values.append(ref_value)
        masks.append(mask)
        ref_embeds.append(ref_embed)
    mask_dict_features = {
        "refs": refs, "ref_values": ref_values, "masks": masks, "ref_embeds": ref_embeds
    }
    return mask_dict_features


# boxinst functions
def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

