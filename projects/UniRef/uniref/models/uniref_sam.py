# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import RobertaModel, RobertaTokenizerFast
from transformers import BertModel, BertTokenizerFast
from transformers import CLIPTextModel, CLIPTokenizerFast

from .segment_anything.build_sam import build_sam_vit_h
from .vos_helper.modules import ValueEncoderSO_Sam
from .fuse_helper.unifusion import UniFusion


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
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
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def compute_mask_iou(inputs, targets):
    """Compute pairwise mask iou between inputs and targets,
    Both have the shape of [N, H*W]
    intputs: mask logits
    targets: torch.float 0/1
    """
    inputs = inputs.sigmoid()
    # thresholding 
    binarized_inputs = (inputs >= 0.5).float() # [N, HW]
    targets = (targets > 0.5).float() # [N, HW]
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


class UniRef_Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        cfg,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        # build model
        self.sam = build_sam_vit_h("pretrained_models/sam_vit_h_4b8939.pth")

        # build reference encoder
        self.with_mask_ref = cfg.MODEL.WITH_MASK_REF
        self.with_lang_ref = cfg.MODEL.WITH_LANG_REF

        # language reference encoder
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
                output_feat_size=self.sam.prompt_encoder.embed_dim,
                dropout=0.1,
                do_ln=True
            )
            self.context_len = cfg.MODEL.LANG_CONFIG.CONTEXT_LEN
        
        # mask reference encoder
        if self.with_mask_ref:
            self.value_encoder = ValueEncoderSO_Sam()

        # unifusion
        self.fusion_module = UniFusion(cfg)

        # freeze sam, only tune mask decoder
        for name, p in self.sam.named_parameters():
            if "mask_decoder" not in name:
                p.requires_grad = False

        # loss weights, follow LISA
        self.bce_loss_weight = 2.0
        self.dice_loss_weight = 0.5

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        images,
        targets: List[Dict],
        lang_dict_features: Dict = None,
        mask_dict_features: Dict = None,
        task: str = "grounding",
        multimask_output: bool = False,
        train: bool = False
    ):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # image encoder
        # ImageList -> Tensor, pad to 1024
        input_images = torch.stack(
            [self.preprocess(image) for image in images], dim=0
        )  # [B, 3, 1024, 1024]
        image_embeddings = self.sam.image_encoder(input_images)         # [B, C, 1024//16, 1024//16]

        # early-fusion
        assert task in ["grounding", "fss", "rvos", "vos"]
        if task == "grounding":
            assert lang_dict_features is not None
            image_embeddings = self.fusion_module([image_embeddings], lang_dict_features=lang_dict_features)[0]
        elif task == "fss" or task == "vos":
            assert mask_dict_features is not None
            image_embeddings = self.fusion_module([image_embeddings], mask_dict_features=mask_dict_features)[0]
        elif task == "rvos":
            assert lang_dict_features is not None
            image_embeddings = self.fusion_module([image_embeddings], lang_dict_features=lang_dict_features, mask_dict_features=mask_dict_features)[0]

        # prompt encoder & mask decoder
        pred_masks = []   # list of tensor, [1, H, W], H,W = 1024
        pred_ious = []    # list of tensor, [1,]
        for i in range(len(image_embeddings)):
            # choose ref embeds as prompt, [1, 1, embed_dim]
            if task in ["grounding", "rvos"]:
                ref_embeds = lang_dict_features["ref_embeds"][i].unsqueeze(0).unsqueeze(0)
            elif task in ["fss", "vos"]:
                ref_embeds = mask_dict_features["ref_embeds"][0][i].unsqueeze(0).unsqueeze(0)
            (
              sparse_embeddings,
              dense_embeddings
            ) = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=ref_embeds
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )  # [1, 1, 256, 256], mask logits; [1, 1], iou scores
            pred_mask = F.interpolate(low_res_masks, scale_factor=4., mode="bilinear", align_corners=False)
            pred_masks.append(pred_mask[:, 0])      # list[tensor], [1, 1024, 1024]
            pred_ious.append(iou_predictions[:, 0]) # list[tensor], [1,]

        outputs = {
            "pred_ious": pred_ious,   # list[tensor]
            "pred_masks": pred_masks  # list[tensor]
        }
        
        # inference
        if not train:
            return outputs, None

        # calculate loss
        mask_bce_loss = 0.
        mask_dice_loss = 0.
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            pred_mask = pred_masks[batch_idx].float()      # [1, H, W]
            gt_mask = targets[batch_idx]["masks"].float()  # [1, H, W]

            assert (
                  gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        loss_dict = {
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
        }
        return outputs, loss_dict

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # x has been normalized
        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


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


