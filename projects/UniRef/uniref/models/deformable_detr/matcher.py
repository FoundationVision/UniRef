# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops
from ...util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        
    def forward_ota(self, outputs, targets):
        """ simOTA for detr
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].sigmoid()
            out_bbox = outputs["pred_boxes"]  # 跳过frame 维度
            indices = []
            matched_ids = []
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx] #[300,4]
                bz_out_prob = out_prob[batch_idx] 
                bz_tgt_ids = targets[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids)
                bz_gtboxs = targets[batch_idx]['boxes'].reshape(num_insts,4) #[num_gt, 4]
                fg_mask, is_in_boxes_and_center  = \
                    self.get_in_boxes_info(bz_boxes,bz_gtboxs,expanded_strides=32)
                pair_wise_ious = ops.box_iou(box_cxcywh_to_xyxy(bz_boxes), box_cxcywh_to_xyxy(bz_gtboxs))
                # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(bz_boxes),  box_cxcywh_to_xyxy(bz_gtboxs))


                # NOTE: cost_class would be 0.
                cost = ( cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center) )  #[num_query,num_gt]
                # import pdb;pdb.set_trace()
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                if bz_gtboxs.shape[0]>0:
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])
                else:
                    indices_i = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_j = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_batchi = (indices_i, indices_j)
                    matched_qidx = []
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        
        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts) #x1y1x2y2
        
        anchor_center_x = boxes[:,0].unsqueeze(1)
        anchor_center_y = boxes[:,1].unsqueeze(1)

        b_l = anchor_center_x > xy_target_gts[:,0].unsqueeze(0)  
        b_r = anchor_center_x < xy_target_gts[:,2].unsqueeze(0)  
        b_t = anchor_center_y > xy_target_gts[:,1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:,3].unsqueeze(0)
        is_in_boxes = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_boxes_all = is_in_boxes.sum(1)>0  # [num_query]

        # in fixed center
        center_radius = 2.5
        b_l = anchor_center_x > (target_gts[:,0]-(1*center_radius/expanded_strides)).unsqueeze(0)  # x1 满足要求
        b_r = anchor_center_x < (target_gts[:,0]+(1*center_radius/expanded_strides)).unsqueeze(0)  # x2 满足要求
        b_t = anchor_center_y > (target_gts[:,1]-(1*center_radius/expanded_strides)).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:,1]+(1*center_radius/expanded_strides)).unsqueeze(0)
        is_in_centers = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_centers_all = is_in_centers.sum(1)>0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all         
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)   

        return is_in_boxes_anchor,is_in_boxes_and_center
    
    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost) # [300,num_gt] 
        ious_in_boxes_matrix = pair_wise_ious
        n_query = len(ious_in_boxes_matrix)
        n_candidate_k = min(n_query, 10)
        
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:,gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:,gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)
        
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) 
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 

        while (matching_matrix.sum(0)==0).any(): 
            num_zero_gt = (matching_matrix.sum(0)==0).sum()
            matched_query_id = matching_matrix.sum(1)>0
            cost[matched_query_id] += 100000.0 
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:,gt_idx])
                matching_matrix[:,gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0: 
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
                matching_matrix[anchor_matching_gt > 1] *= 0 
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 

        assert not (matching_matrix.sum(0)==0).any() 
        # 
        selected_query = matching_matrix.sum(1)>0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix==0] = cost[matching_matrix==0] + float('inf')
        matched_query_id = torch.min(cost,dim=0)[1]

        # in this version, the selected_query is [300,] with true/false
        # we convert it to value indices
        matched_anchor_inds = torch.arange(len(matching_matrix)).to(gt_indices)
        selected_query = matched_anchor_inds[selected_query]     # [num_pos,]

        return (selected_query,gt_indices) , matched_query_id

    def forward_ori(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # [batch_size * num_queries, K]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
