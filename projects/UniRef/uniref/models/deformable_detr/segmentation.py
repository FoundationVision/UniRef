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

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from detectron2.structures import Instances



class DETRsegm_boxonly(nn.Module):
    def __init__(self, detr, ota=False):
        super().__init__()
        self.debug_only = False
        self.detr = detr
        self.ota = ota
        self.max_insts_num = 100

    def coco_forward(self, samples, gt_targets, criterion, train=False):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        if self.debug_only:
            self.debug_data(samples, gt_targets)
        outputs = self.detr(samples)
        loss_dict = criterion(outputs, gt_targets)

        return outputs, loss_dict


    def coco_inference(self, samples, gt_targets, criterion, train=False):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        if self.debug_only:
            self.debug_data(samples, gt_targets)
        outputs = self.detr(samples)
        # loss_dict = criterion(outputs, gt_targets)

        return outputs, None


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



def segmentation_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
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

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        # import pdb;pdb.set_trace()
        mask = F.interpolate(results.pred_masks.float(), size=(output_height, output_width), mode='nearest')
        # import pdb;pdb.set_trace()
        mask = mask.squeeze(1).byte()
        results.pred_masks = mask

        # import pdb;pdb.set_trace()
        #  results.pred_masks [N, output-height, output-width]


    return results


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


def mask_iou_loss(inputs, targets, num_boxes, eps=1e-7, size_average=True):
    """
    Compute the mask iou loss, similar to IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # [all_num_insts, H, W]
    assert len(inputs.shape) == 3 and inputs.shape == targets.shape

    inter = torch.min(inputs, targets).sum(2).sum(1)
    union = torch.max(inputs, targets).sum(2).sum(1)

    if size_average:
        iou = torch.sum((inter+eps) / (union+eps)) / num_boxes
    else:
        iou = (inter+eps) / (union+eps)
    
    loss = 1.0 - iou
    return loss

def bootstrap_ce_loss(inputs, targets, num_boxes, bootstrap=0.4):
    """
    Compute the mask iou loss, similar to IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    N, _, H, W = inputs.shape

    inputs = -1.0 * torch.log(inputs)

    if bootstrap > 0:
        num = int(H * W * bootstrap)
        loss = torch.sum(inputs * targets, dim=1).view(N, -1)
        mloss, _ = torch.sort(loss, dim=-1, descending=True)
        loss = torch.mean(mloss[:, :num])
    else:
        loss = torch.sum(inputs * targets)
        loss = loss.mean()
    return loss



# soft-aggregation 
def soft_aggregate(prob, keep_bg=False):
    # prob: [K, H, W], the mask prob scores of K objects
    k = prob.shape
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]
