import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

from ...util import box_ops
from .segmentation import (dice_loss, sigmoid_focal_loss, mask_iou_loss, bootstrap_ce_loss, soft_aggregate)
from ...util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from fvcore.nn import giou_loss, smooth_l1_loss



class SmoothLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1, ignore_index=None):
        super().__init__()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')

        self.ignore_index = ignore_index

    def forward(self, feature, target):
        feature = feature.float()
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target[valid_mask]
            feature = feature[valid_mask]
        assert target.numel() > 0
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        loss = self.kl(log_prb, one_hot)
        return loss.sum(dim=1).mean()



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 mask_out_stride=4, num_frames=1, ota=False, mask_aux_loss=False, cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.num_frames = num_frames
        self.ota = ota
        self.mask_aux_loss = mask_aux_loss
        # boxinst configs
        if cfg is not None:
            self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
            self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
            self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
            self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
            self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
            self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
            self.boxinst_topk = cfg.MODEL.BOXINST.TOPK
            self.register_buffer("_iter", torch.zeros([1]))

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [B, N, K]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if len(target_classes_o) == 0: # no gt in the batch
            loss_ce = src_logits.sum() * 0.0
            losses = {'loss_ce': loss_ce}
            return losses

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        num_boxes = len(idx[0]) if self.ota else num_boxes
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        use_iou_branch = "pred_boxious" in outputs

        if len(target_boxes) == 0:
            losses = {}
            losses['loss_bbox'] = src_boxes.sum() * 0.0
            losses['loss_giou'] = src_boxes.sum() * 0.0
            if use_iou_branch:
                losses['loss_boxiou'] = src_boxes.sum() * 0.0
            return losses

        # box iou
        if use_iou_branch:
            with torch.no_grad():
                ious = compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes),
                                        box_ops.box_cxcywh_to_xyxy(target_boxes))                    
            tgt_iou_scores = ious
            src_iou_scores = outputs['pred_boxious'] # [B, N, 1]
            src_iou_scores = src_iou_scores[idx]
            src_iou_scores = src_iou_scores.flatten(0)
            tgt_iou_scores = tgt_iou_scores.flatten(0)
            loss_boxiou = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')


        num_boxes = src_boxes.shape[0] if self.ota else num_boxes
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes),box_ops.box_cxcywh_to_xyxy(target_boxes))
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        if use_iou_branch:
            losses['loss_boxiou'] = loss_boxiou
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        
        src_idx = self._get_src_permutation_idx(indices) 
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] # list[tensor]: bs x [1, num_inst, num_frames, H/4, W/4]
        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        if type(src_masks) == list:
            src_masks = torch.cat(src_masks, dim=1)[0]  # [num_all_inst, num_frames, H/4, W/4]
        if src_masks.ndim == 0:
            # no mask label (only box label)
            losses = {}
            losses['loss_mask'] = src_masks * 0.0
            losses['loss_dice'] = src_masks * 0.0
            if self.mask_aux_loss:
                losses["loss_mask_cls"] = src_masks * 0.0
                losses["loss_mask_iou"] = src_masks * 0.0
            return losses

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                            size_divisibility=32,
                                                            split=False).decompose()
        # during training, the size_divisibility is 32
        target_masks = target_masks.to(src_masks) # [bs, max_num_gt, H, W]
        # for VOS, supervised in the original resolution
        if target_masks.shape[-2:] == src_masks.shape[-2:]:
            pass
        else:
            # downsample ground truth masks with ratio mask_out_stride
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w
        num_frames = src_masks.shape[1]
        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx] # [num_all_inst, num_frames, H/4, W/4]

        num_boxes = src_masks.shape[0] if self.ota else num_boxes

        if len(target_masks) == 0: # no gt
            losses = {}
            losses['loss_mask'] = src_masks.sum() * 0.0
            losses['loss_dice'] = src_masks.sum() * 0.0

            if self.mask_aux_loss:
                losses["loss_mask_cls"] = src_masks.sum() * 0.0
                losses["loss_mask_iou"] = src_masks.sum() * 0.0
            return losses


        if self.mask_aux_loss:
            # convert instance mask to semantice mask
            src_masks_aux = src_masks.flatten(0,1).sigmoid() # scores
            src_masks_aux_list = []
            for src_mask_aux in src_masks_aux:
                src_masks_aux_list.append(soft_aggregate(src_mask_aux.unsqueeze(0), keep_bg=True))
            src_masks_aux = torch.stack(src_masks_aux_list, dim=0) # [all_num_insts, 2, H, W]
            # convert targets, including bg
            target_masks_aux = torch.zeros_like(src_masks_aux)    # [all_num_insts, 2, H, W]
            target_masks_aux[:, 1, :, :] = target_masks.flatten(0,1).float()
            target_masks_aux[:, 0, :, :] = 1.0 - target_masks.flatten(0,1).float()

        src_masks = src_masks.flatten(1)  
        target_masks = target_masks.flatten(1)
        # src_masks/target_masks: [n_targets, num_frames * H * W]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

        if self.mask_aux_loss:
            losses.update({"loss_mask_cls": bootstrap_ce_loss(src_masks_aux, target_masks_aux, num_boxes)})
            losses.update({"loss_mask_iou": mask_iou_loss(src_masks_aux[:, 1, :, :], target_masks_aux[:, 1, :, :], num_boxes)})

        return losses

    def loss_masks_boxinst(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        self._iter += 1
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"] # list[tensor]: bs x [1, num_inst, num_frames, H/4, W/4]
        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        if type(src_masks) == list:
            src_masks = torch.cat(src_masks, dim=1)[0]  # [num_all_inst, num_frames, H/4, W/4]
        if src_masks.ndim == 0:
            # no mask label (only box label)
            losses = {}
            losses['loss_prj'] = src_masks * 0.0
            losses['loss_pairwise'] = src_masks * 0.0
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # pick part of samples to compute loss because BoxInst consumes more memory
        if len(tgt_idx[0]) > self.boxinst_topk:
            keep_indexs = random.sample(range(len(tgt_idx[0])), self.boxinst_topk)
            src_idx = tuple([src_idx[0][keep_indexs], src_idx[1][keep_indexs]])
            tgt_idx = tuple([tgt_idx[0][keep_indexs], tgt_idx[1][keep_indexs]])
            src_masks = src_masks[keep_indexs]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = self.get_target_masks(targets, src_masks)
        num_frames = src_masks.shape[1]
        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx] # [num_all_inst, num_frames, H/4, W/4]

        # num_boxes = src_masks.shape[0] if self.ota else num_boxes

        if len(target_masks) == 0: # no gt
            losses = {}
            losses['loss_prj'] = src_masks.sum() * 0.0
            losses['loss_pairwise'] = src_masks.sum() * 0.0
            return losses
        
        # box-supervised BoxInst losses
        mask_scores = src_masks.sigmoid()

        image_color_similarity_list = []
        for (batch_idx, target_idx) in zip(tgt_idx[0], tgt_idx[1]):
            image_color_similarity_list.append(targets[batch_idx]["image_color_similarity"][target_idx])
        image_color_similarity = torch.stack(image_color_similarity_list, dim=0).to(dtype=mask_scores.dtype) # (N, 8, H//4, W//4)

        loss_prj_term = compute_project_term(mask_scores, target_masks)

        pairwise_losses = compute_pairwise_term(
            src_masks, self.pairwise_size,
            self.pairwise_dilation
        )

        weights = (image_color_similarity >= self.pairwise_color_thresh).float() * target_masks.float() # (N, 8, H//4, W//4)
        loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

        warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
        loss_pairwise = loss_pairwise * warmup_factor

        losses = {
            "loss_prj": loss_prj_term,
            "loss_pairwise": loss_pairwise,
        }

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'masks_boxinst': self.loss_masks_boxinst,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, indices_list):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices_list[-1], num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                indices = indices_list[i]
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # two-stage
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            # hack implementation for Hungarian Matcher in encoder
            indices = self.matcher.forward_ori(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'masks_boxinst']:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
    
    def get_target_masks(self, targets, src_masks):
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                            size_divisibility=32,
                                                            split=False).decompose()
        target_masks = target_masks.to(src_masks)
        # downsample ground truth masks with ratio mask_out_stride
        if self.mask_out_stride != 1:
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w
        return target_masks


def compute_box_iou(inputs, targets):
    """Compute pairwise iou between inputs, targets
    Both have the shape of [N, 4] and xyxy format
    """
    area1 = box_ops.box_area(inputs)
    area2 = box_ops.box_area(targets)

    lt = torch.max(inputs[:, None, :2], targets[:, :2])  # [N,M,2]
    rb = torch.min(inputs[:, None, 2:], targets[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)        # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    iou = torch.diag(iou)
    return iou

# boxinst
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

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss