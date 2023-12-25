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
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from fvcore.nn import giou_loss, smooth_l1_loss

from .models.uniref_sam import UniRef_Sam

from .models.ddetrs import segmentation_postprocess
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
import torchvision.ops as ops

from einops import repeat
import os
from PIL import Image
from skimage import color

# for visualization
from detectron2.structures import BoxMode
import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

__all__ = ["UniRef_SAM"]


@META_ARCH_REGISTRY.register()
class UniRef_SAM(nn.Module):
    """
    Implement UniRef_SAM
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM  # for image, is 1 
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Task parameters
        self.nshot = cfg.TASK.FSS.NSHOT

        # Model
        self.model = UniRef_Sam(cfg)
        self.mask_thres = 0.5


        # -------------------------------------------------------------------------
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.use_lsj = cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj"

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
        if task in ["grounding", "fss"]:
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
            # image tasks
            if task in ["grounding", "fss"]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # video tasks
            else:
                gt_instances = []
                for video in batched_inputs:
                    for frame in video["instances"]:
                        gt_instances.append(frame.to(self.device))
            targets = self.prepare_targets(gt_instances)

            if task == "grounding":
                assert self.num_frames == 1
                captions = [x["expressions"] for x in batched_inputs] # list[str]
                lang_dict_features = self.forward_text(captions, device="cuda")
                output, loss_dict = self.model.forward(images, targets, train=True,
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
                output, loss_dict = self.model.forward(images, targets, train=True,
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
                output, loss_dict = self.model.forward(det_images, det_targets, train=True,
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
                    output, loss_dict = self.model.forward(det_images, det_targets, train=True, 
                                            lang_dict_features=lang_dict_features, task=task)
                # rvos data
                else:  
                    if 0. < torch.rand(1) < 0.4:  # only fuse with language
                        output, loss_dict = self.model.forward(det_images, det_targets, train=True, 
                                                lang_dict_features=lang_dict_features, task=task)
                    else:  # fuse with language and mask
                        ref_images = ImageList.from_tensors([images[id] for id in ref_ids])
                        ref_gt_instances = [gt_instances[id] for id in ref_ids]
                        ref_targets = self.prepare_targets(ref_gt_instances)
                        ref_masks = [target["masks"] for target in ref_targets]  # list[tensor], length of bs, [1, H, W], image size
                        # forward to get the mask_dict_features
                        srcs_ref, masks_ref, _, _ = self.forward_ref_backbone(ref_images)  # multi-scale features, 8x -> 64x
                        mask_dict_features = self.forward_template(ref_images, ref_masks, srcs_ref, masks_ref)
                        output, loss_dict = self.model.forward(det_images, det_targets, train=True,
                                    lang_dict_features=lang_dict_features, mask_dict_features=mask_dict_features, task=task)

            return loss_dict

        # -----------------------------------------------------------------------------------------
        # inference
        else:
            # image task
            if task in ["grounding", "fss"]:
                if task == "grounding":
                    captions = [x["expressions"] for x in batched_inputs]  # list[str]
                    lang_dict_features = self.forward_text(captions, device="cuda")
                    output, _ = self.model.forward(images, None, train=False,
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
                    output, _ = self.model.forward(images, None, train=False,
                                        mask_dict_features=mask_dict_features, task=task) 

                # image task (coco, refcoco) post-process
                iou_pred = output["pred_ious"]    # list of [1,]
                mask_pred = output["pred_masks"]  # list of [1, 1024, 1024]
                results = self.inference(iou_pred, mask_pred, images.image_sizes)

                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    # here is the original image size
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width, filter_empty=False)
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
                        self.inference_rvos_vl(batched_inputs, images, aggregate_objects=True)
                return

    def prepare_targets(self, targets):
        # padding gt_masks to max size over a batch (This is important for training with propagation)
        if hasattr(targets[0], "gt_masks"):
            # mask size: (n_inst, hm, wm)
            gt_masks_list = [x.gt_masks if self.use_lsj else x.gt_masks.tensor for x in targets]
            max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
            stride = 1024 # size_divisibility for SAM
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride

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
                stride = 1024 # size_divisibility for SAM
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
                output, _ = self.model.forward(images, None, train=False,
                                mask_dict_features=cur_mask_dict_features, task=task)
                iou_pred = output["pred_ious"]   # iou score
                mask_pred = output["pred_masks"] # mask logit
                # loop over  each image, online fashion for 1 time
                for _, (iou, output_mask, image_size) in enumerate(zip(
                    iou_pred, mask_pred, images.image_sizes
                )):
                    topk_values = iou  # [1,]
                    if topk_values > score_thres:
                        track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
                        mask_dict[obj_id] = track_masks[0, 0]
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
                    samples = nested_tensor_from_tensor_list(images, size_divisibility=1024)
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
                    outputs, _ = self.model.forward(clip_images, None, train=False, 
                                    lang_dict_features=lang_dict_features, task=task)
                
                    # inference each frame independently
                    video_ious = outputs["pred_ious"]
                    video_output_masks = outputs["pred_masks"]
                    # loop over each image
                    for _, (iou, output_mask, image_size) in enumerate(zip(
                        video_ious, video_output_masks, images.image_sizes
                    )): 
                        topk_values = iou
                        if topk_values > score_thres:
                            track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
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
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=1024)
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
                    outputs, _ = self.model.forward(clip_images, None, train=False, 
                                    lang_dict_features=lang_dict_features, mask_dict_features=cur_mask_dict_features, task=task)

                    # inference each frame independently
                    video_ious = outputs["pred_ious"]
                    video_output_masks = outputs["pred_masks"]
                    # loop over each image
                    for _, (iou, output_mask, image_size) in enumerate(zip(
                        video_ious, video_output_masks, images.image_sizes
                    )): 
                        topk_values = iou
                        if topk_values > score_thres:
                            track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
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
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=1024)
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
                        outputs, _ = self.model.forward(clip_images, None, train=False, 
                                            lang_dict_features=lang_dict_features, task=task)
                        # inference each frame independently
                        video_ious = outputs["pred_ious"]
                        video_output_masks = outputs["pred_masks"]
                        # loop over each image, online fashion only for 1 time
                        for _, (iou, output_mask, image_size) in enumerate(zip(
                            video_ious, video_output_masks, images.image_sizes
                        )): 
                            track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
                            track_masks = track_masks[0, 0]
                            final_masks.append(track_masks)
                        del outputs

                        # store the masks to ref_init_dict and ref_prev_dict
                        cur_mask = track_masks > 0.5
                        # davis17, all objects appear in the first frame
                        # from origin image size to image size
                        samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=1024)
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
                        outputs, _ = self.model.forward(clip_images, None, train=False, 
                                        lang_dict_features=lang_dict_features, mask_dict_features=cur_mask_dict_features, task=task)

                        # inference each frame independently
                        video_ious = outputs["pred_ious"]
                        video_output_masks = outputs["pred_masks"]
                        # loop over each image
                        for _, (iou, output_mask, image_size) in enumerate(zip(
                            video_ious, video_output_masks, images.image_sizes
                        )): 
                            track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                            track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
                            track_masks = track_masks[0, 0]
                            final_masks.append(track_masks)
                        del outputs

                        # step 3: update ref_prev_dict
                        cur_mask = track_masks > 0.5 # binary mask
                        if (cur_mask > 0).any(): 
                            # from origin image size to image size
                            samples = nested_tensor_from_tensor_list(clip_images, size_divisibility=1024)
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
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            lang_dict_features = self.forward_text(captions, device="cuda")

            # get the final_masks
            final_masks = []
            # inference for each frame
            for frame_idx in range(video_length):
                clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                clip_images = self.preprocess_clip_image(clip_inputs)
                outputs, _ = self.model.forward(clip_images, None, train=False, 
                                    lang_dict_features=lang_dict_features, task=task)
            
                # inference each frame independently
                video_ious = outputs["pred_ious"]
                video_output_masks = outputs["pred_masks"]
                # loop over each image
                for _, (iou, output_mask, image_size) in enumerate(zip(
                    video_ious, video_output_masks, images.image_sizes
                )): 
                    topk_values = iou
                    if topk_values > score_thres:
                        track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
                        track_masks = track_masks[0, 0]
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
                # captions: list[str]
                captions = [expressions[exp_id]]
                lang_dict_features = self.forward_text(captions, device="cuda")
                # 2. for each frame
                for frame_idx in range(video_length):
                    clip_inputs = [{"image": batched_inputs[0]["image"][frame_idx:frame_idx+1]}]
                    clip_images = self.preprocess_clip_image(clip_inputs)
                    outputs, _ = self.model.forward(clip_images, None, train=False, 
                                    lang_dict_features=lang_dict_features, task=task)

                    # inference each frame independently
                    video_ious = outputs["pred_ious"]
                    video_output_masks = outputs["pred_masks"]
                    # loop over each image, online fashion only for 1 time
                    for _, (iou, output_mask, image_size) in enumerate(zip(
                        video_ious, video_output_masks, images.image_sizes
                    )): 
                        track_masks = output_mask[:, None, :image_size[0], :image_size[1]]  # [1, 1, H, W], crop the padding area
                        track_masks = F.interpolate(track_masks, size=(height, width), mode='bilinear', align_corners=False) # [1, 1, ori_h, ori_w]
                        track_masks = track_masks[0, 0]
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

    # inference for image tasks, SAM selects the top-1
    def inference(self, iou_pred, mask_pred, image_sizes):
        """Arguments:
            iou_pred (list[tensor]): list of [1,]
            mask_pred (list[tensor]): list of [1, H, W] in a batch, H,W = 1024.
            image_sizes (list[torch.Size]): the input image sizes
        """
        assert len(mask_pred) == len(image_sizes)
        results = []

        for i, (iou_pred_per_image, mask_pred_per_image, image_size) in enumerate(zip(iou_pred, mask_pred, image_sizes)):
            result = Instances(image_size)
            # remove padding and to binary mask
            mask_pred_per_image = mask_pred_per_image[:, None, :image_size[0], :image_size[1]]  # [1, 1, h, w]
            # result.pred_boxes = Boxes(torch.zeros(mask_pred_per_image.size(0), 4))
            # box generate from mask, uncomment the following line
            result.pred_boxes = BitMasks(mask_pred_per_image[:, 0] > 0).get_bounding_boxes()
            mask_pred_per_image = mask_pred_per_image.sigmoid() > self.mask_thres 
            result.pred_masks = mask_pred_per_image
            result.scores = iou_pred_per_image
            result.pred_classes = torch.zeros((1,), dtype=torch.long, device=mask_pred_per_image.device)
            results.append(result)
        return results

    def preprocess_support_images(self, support_images):
        images = [self.normalizer(image.to(self.device)) for image in support_images]
        images = ImageList.from_tensors(images, size_divisibility=1024)
        return images

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, size_divisibility=1024)  # make image size (1024, 1024)
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
            images = ImageList.from_tensors(images, size_divisibility=1024)
        else:
            images = []
            for video in batched_inputs:
                for idx in clip_idx:
                    images.append(self.normalizer(video["image"][idx].to(self.device)))
            images = ImageList.from_tensors(images, size_divisibility=1024)
        return images

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.model.tokenizer.batch_encode_plus(
                captions, padding="max_length", truncation=True, max_length=self.model.context_len, return_tensors="pt").to(device)
            encoded_text = self.model.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length], padding positions are 1

            text_features = encoded_text.last_hidden_state 
            text_features = self.model.resizer(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            if self.model.lang_pool:
                text_word_features, text_word_masks = text_features.decompose()
                embedded = text_word_features * text_word_masks.ne(1).unsqueeze(-1) # [batch_size, length, C]
                text_sentence_features = embedded.sum(1) / (text_word_masks.ne(1).sum(-1).unsqueeze(-1).float()) # [batch_size, C]
            else:
                text_sentence_features = encoded_text.pooler_output  
                text_sentence_features = self.model.resizer(text_sentence_features)  

            lang_dict_features = {}
            lang_dict_features["refs"] = text_word_features       # [bs, seq, c]
            lang_dict_features["ref_values"] = text_word_features # [bs, seq, c]
            lang_dict_features["masks"] = text_word_masks         # [bs, seq], bool, padding locations are 1
            lang_dict_features["ref_embeds"] = text_sentence_features # [bs, c]
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return lang_dict_features


    def forward_ref_backbone(self, samples):
        # samples (ImageList)
        input_images = torch.stack(
            [self.model.preprocess(image) for image in samples], dim=0
        )  # [B, 3, 1024, 1024]
        image_embeddings = self.model.sam.image_encoder(input_images)    # [B, C, 1024//16, 1024//16]
        n, c, h, w = image_embeddings.shape
        srcs = [image_embeddings]

        image_sizes = samples.image_sizes  # before padding
        masks = torch.ones((n, h, w), dtype=torch.bool, device=image_embeddings.device)
        for i, image_size in enumerate(image_sizes):
            masks[i][:image_size[0], :image_size[1]] = False
        masks = [masks]
        return srcs, masks, None, None


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
        samples = torch.stack(
            [self.model.preprocess(image) for image in samples], dim=0
        )  # [B, 3, 1024, 1024]
        
        # ref_tensor_list, ref_mask_list = [], []
        gt_mask_list = []
        for i in range(len(samples)):   # for B times
            # get mask in image size, [1, 1, H, W]
            gt_mask = masks[i][None]
            gt_mask_list.append(gt_mask)
        gt_masks = torch.cat(gt_mask_list, dim=0)       # [B, 1, H, W]
    
        # forward to get the reference frame multi-scale values, play as V
        values_ref = self.model.value_encoder(samples, gt_masks, srcs_ref)

        refs = []
        masks = []
        ref_values = [] 
        ref_embeds = []
        for n_l in range(1):  # one-scale feature for SAM
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

        



# --------------------------------------------------------------------------------
# debug visualiza purpose
import numpy as np
import cv2

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def debug_template_4c(samples):
    import numpy as np
    import cv2
    import torch.distributed as dist
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    for i in range(len(samples.tensors)):
        image_mask = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() # (H, W, 4)
        image = image_mask[:, :, :3]
        image = image * std + mean # (H, W, 3)
        gt_mask = image_mask[:, :, -1] # (H, W)
        input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        image[:, :, -1] = np.clip(image[:, :, -1] + 100 * gt_mask, 0, 255)
        cv2.imwrite("rank_0_batch_%d_template_img.jpg"%(i), image)
        cv2.imwrite("rank_0batch_%d_template_mask.jpg"%(i), input_mask)

def debug_data(images, targets):
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    for i in range(len(targets)):
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        target = targets[i]
        boxes = target["boxes"].cpu().numpy()
        num_inst = boxes.shape[0]
        for j in range(num_inst):
            cx, cy, w, h = boxes[j] * target["image_size"].cpu().numpy() # image size without padding
            x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
            if "masks" in target:
                mask = target["masks"][j].cpu().float().numpy() # (H, W)
                if mask.shape != image.shape[:-1]:  # mask has padding while image not
                    ori_h, ori_w = image.shape[:2]
                    mask_new = np.zeros((image.shape[:-1]))
                    mask_new[:ori_h, :ori_w] = mask[:ori_h, :ori_w]
                else:
                    mask_new = mask
                image[:, :, -1] += 128 * mask_new
            if "inst_id" in target and target["inst_id"][j] != -1:
                color = color_list[target["inst_id"][j] % num_color]
            else:
                color = default_color
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
    return 



def vis_add_mask(img, mask, color):
    # visaulize one mask
    # origin_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    origin_img = img

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')

    alpha = 0.5
    for c in range(3):
        origin_img[:, :, c] = np.where(mask == 1,
                origin_img[:, :, c] * (1 - alpha) + alpha * color[c],
                origin_img[:, :, c])
    return origin_img


def show_images(images, targets=None, title=''):
    # images (tensor): [B, C, H, W]
    image = images[0]
    h, w = image.shape[-2:]

    color_list = colormap()
    # color = np.array([1.000, 0.000, 0.000]).astype(np.float32) * 255
    PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1) # RGB
    PIXEL_STD = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

    img = image
    normed_img = img.to("cpu")
    resize_img = normed_img * PIXEL_STD + PIXEL_MEAN
    resize_img = resize_img[-torch.arange(3) + 2,:,:].permute(1, 2, 0).float().to("cpu").numpy()
    resize_img = resize_img.copy()

    if targets is not None:
        target = targets[0]
        gt_boxes = target["boxes"]  # cxcywh in [0, 1]
        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        image_size = torch.as_tensor([w, h, w, h], dtype=torch.float, device=image.device)
        gt_boxes = gt_boxes * image_size
        # gt_boxes = target.gt_boxes.tensor # xyxy in image size
        # gt_scores = target.similarities if target.has('similarities') else None
        mask_on = "masks" in target
        if mask_on:
            gt_masks = target["masks"].cpu().numpy()

        for n in range(len(gt_boxes)):
            # add boxes
            boxes = gt_boxes[n].tolist()
            cv2.rectangle(resize_img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), color=color_list[n%79].tolist(), thickness=2)
            # if gt_scores is not None:
            #     score = gt_scores[n].tolist()
            #     score = 1 - score
            #     cv2.putText(resize_img, "{:.2f}".format(score), (int(boxes[0]), int(boxes[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=color.tolist(), thickness=2)
            if mask_on:
                resize_img = vis_add_mask(resize_img, gt_masks[n], color=color_list[n%79])

    if title == '':
        title = 'test'
    cv2.imwrite(title + '.png', resize_img)
    return

# for inference
def show_predictions(images, targets, captions=None, title=''):
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    for i in range(len(targets)):
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        target = targets[i]
        scores = target.scores
        choose = scores > 0.5
        target = target[choose.cpu()]
        boxes = target.pred_boxes.tensor.cpu().numpy()  # [N, 4], in image size
        num_inst = boxes.shape[0]
        if captions is not None:
            caption = captions[i]
            cv2.putText(image, "{}".format(caption), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=default_color, thickness=2)
        for j in range(num_inst):
            x1, y1, x2, y2 = boxes[j]
            color = color_list[j % num_color]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            if target.has("pred_masks"):
                mask = target.pred_masks[j][0].cpu().float().numpy() # [H, W]
                if mask.shape != image.shape[:-1]:  # mask has padding while image not
                    ori_h, ori_w = image.shape[:2]
                    mask_new = np.zeros((image.shape[:-1]))
                    mask_new[:ori_h, :ori_w] = mask[:ori_h, :ori_w]
                else:
                    mask_new = mask
                image = vis_add_mask(image, mask_new, color=color)
            if target.has("scores"):
                score = target.scores[j].cpu().numpy()
                cv2.putText(image, "{:.2f}".format(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=color, thickness=2)
        if title == '':
            title = 'test'
        cv2.imwrite(title + '.png', image)
    return 
