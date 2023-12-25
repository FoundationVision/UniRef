# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from distutils.command.build import build
import logging

import numpy as np
import torch
import random
import os
import json
from detectron2 import data

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures.boxes import BoxMode
import re

from fvcore.transforms.transform import HFlipTransform
from pycocotools.coco import COCO

__all__ = ["SamDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeLongestEdge(1024))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


# SAM receives the input image of fixed size (1024, 1024)
class SamDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        # for fss
        self.nshot = cfg.TASK.FSS.NSHOT

        # oridinal numbers
        self.ordinal_nums = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

    def has_ordinal_num(self, expressions_list):
        flag = False
        for expression in expressions_list:
            expression_low = expression.lower()
            for word in self.ordinal_nums:
                if word in expression_low:
                    flag = True
                    break
            if flag == True:
                break
        return flag


    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        expression = expressions[np.random.choice(len(expressions))]
        expression = clean_string(expression)
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

    def load_support_images_for_fss(self, dataset_dict, tfm_gens, disable_crop, support_image_id):
        flag = False
        while not flag:
            split = dataset_dict["split"]
            # NOTE: the images and annos must strictly corresponds 1-to-1 with id，the json file has been preprocessed
            gt = json.load(open(f'./datasets/fss-1000/annotations/{split}.json', 'r'))
            gt_imgs, gt_anns = gt["images"], gt["annotations"]

            # load support image
            support_image_dict = gt_imgs[support_image_id - 1]
            assert support_image_dict['id'] == support_image_id
            support_image_name = support_image_dict['file_name']
            support_image_path = os.path.join('./datasets/fss-1000/images', support_image_name)
            support_image = utils.read_image(support_image_path, format=self.img_format)
            if self.crop_gen is None or disable_crop:
                support_image, support_transforms = T.apply_transform_gens(tfm_gens, support_image)
            else:
                if np.random.rand() > 0.5:
                    support_image, support_transforms = T.apply_transform_gens(tfm_gens, support_image)
                else:
                    support_image, support_transforms = T.apply_transform_gens(
                        tfm_gens[:-1] + self.crop_gen + tfm_gens[-1:], support_image
                    )
            support_image_shape = support_image.shape[:2]  # h, w
            support_image = torch.as_tensor(np.ascontiguousarray(support_image.transpose(2, 0, 1)))

            # load support image annotation
            support_anno_dict = gt_anns[support_image_id - 1]
            assert support_anno_dict['image_id'] == support_image_id
            support_annotations = [support_anno_dict]
            for anno in support_annotations:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
                anno["bbox_mode"] = BoxMode.XYWH_ABS
            # USER: Implement additional transformations if you have other types of data
            support_annos = [
                utils.transform_instance_annotations(obj, support_transforms, support_image_shape)
                for obj in support_annotations
                if obj.get("iscrowd", 0) == 0
            ]
            support_instances = utils.annotations_to_instances(support_annos, support_image_shape, mask_format="bitmask")
            if hasattr(support_instances, "gt_masks"):
                support_instances.gt_boxes = support_instances.gt_masks.get_bounding_boxes()
            support_instances = utils.filter_empty_instances(support_instances)
            if len(support_instances) != 0:
                flag = True
        return support_image, support_instances


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # if there are ordinal numbers in expressions, disable crop
        disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False

        # augmentation use low-resolution for fss tasks
        task = dataset_dict["task"]
        tfm_gens = self.tfm_gens

        if self.crop_gen is None or disable_crop:
            image, transforms = T.apply_transform_gens(tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    tfm_gens[:-1] + self.crop_gen + tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if "expressions" in dataset_dict:
            # pick one expression if there are multiple expressions during training,
            # for validation, one image only has one expression in refcoco.
            dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)
        
        # ==========================================================================================
        # fss, load support images/masks
        if task == "fss":
            assert dataset_dict["dataset_name"] == "fss", "We only supports fss-1000 dataset now."
            nshot = 1 if self.is_train else self.nshot
            support_image_ids = random.sample(dataset_dict["support_image_ids"], nshot) if self.is_train else dataset_dict["support_image_ids"][:nshot]
            support_images, support_instances = [], []  
            for support_image_id in support_image_ids:
                support_image, support_instance = self.load_support_images_for_fss(dataset_dict, tfm_gens, disable_crop, support_image_id)
                support_images.append(support_image)
                support_instances.append(support_instance)
            dataset_dict["support_images"] = support_images        # list[Tensor]
            dataset_dict["support_instances"] = support_instances  # list[Instance]
        # ==========================================================================================

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            if hasattr(instances, "gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            if len(instances) == 0:
                return None 
            dataset_dict["instances"] = instances

        
        return dataset_dict

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')