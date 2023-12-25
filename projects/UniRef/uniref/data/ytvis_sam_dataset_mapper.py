import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
import re

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation, build_augmentation_sam

from fvcore.transforms.transform import HFlipTransform

__all__ = ["YTVISSamDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISSamDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_range_vos: int = 100,
        sampling_frame_range_rvos: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        multidataset: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.multidataset           = multidataset
        if self.multidataset:
            self.augmentations              = [T.AugmentationList(x) for x in augmentations]
            self.augmentations_nocrop       = [T.AugmentationList(x) if x is not None else None for x in augmentations_nocrop]
        else:
            self.augmentations              = T.AugmentationList(augmentations)
            if augmentations_nocrop is not None:
                self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
            else:
                self.augmentations_nocrop   = None
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_range_vos  = sampling_frame_range_vos
        self.sampling_frame_range_rvos = sampling_frame_range_rvos
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler" and is_train:
            multidataset = True
            # different support for different image sizes
            # min_size, max_size not usd actually, as SAM uses fixed size of (1024, 1024)
            assert len(cfg.INPUT.MIN_SIZE_TRAIN_MULTI) == len(cfg.INPUT.MAX_SIZE_TRAIN_MULTI)
            augs_nocrop, augs = [], []
            for (min_size_train, max_size_train) in zip(cfg.INPUT.MIN_SIZE_TRAIN_MULTI, cfg.INPUT.MAX_SIZE_TRAIN_MULTI):
                if cfg.INPUT.CROP.ENABLED and is_train:
                    augs_nocrop_cur, augs_cur = build_augmentation_sam(cfg, is_train, min_size_train, max_size_train)
                else:
                    augs_cur = build_augmentation_sam(cfg, is_train, min_size_train, max_size_train)
                    augs_nocrop_cur = None
                augs_nocrop.append(augs_nocrop_cur)
                augs.append(augs_cur)
        else:
            multidataset = False
            if cfg.INPUT.CROP.ENABLED and is_train:
                augs_nocrop, augs = build_augmentation_sam(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
            else:
                augs = build_augmentation_sam(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
                augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_range_vos": cfg.INPUT.SAMPLING_FRAME_RANGE_VOS,
            "sampling_frame_range_rvos": cfg.INPUT.SAMPLING_FRAME_RANGE_RVOS,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.DDETRS.NUM_CLASSES,
            "multidataset": multidataset
        }

        return ret

    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        if self.is_train:
            expression = expressions[np.random.choice(len(expressions))]
            expression = clean_string(expression)
        else:
            if isinstance(expressions[0], list):
                # for refdavis, the json has been preprocessed
                # so "expressions": [["exp1", "exp2", ...]]
                expression = [clean_string(e) for e in expressions[0]]  # list
            else:
                # for refcoco and refytvos, the json has been preprocessed
                # so only one "expressions": ["exp1"]
                expression = expressions[0]
                expression = clean_string(expression)                   # str
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            if isinstance(expression, list):
                expression = [e.replace('left', '@').replace('right', 'left').replace('@', 'right') for e in expression]
            else:
                expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            if video_length == 1:
                # get pseudo video from static image
                dataset_dict["file_names"] = dataset_dict["file_names"] * self.sampling_frame_num
                dataset_dict["annotations"] = dataset_dict["annotations"] * self.sampling_frame_num
                video_length = self.sampling_frame_num
                
            ref_frame = random.randrange(video_length)

            if dataset_dict["task"] == "vos" and self.sampling_frame_range_vos is not None:
                sampling_frame_range = self.sampling_frame_range_vos
            elif dataset_dict["task"] == "rvos" and self.sampling_frame_range_rvos is not None:
                sampling_frame_range = self.sampling_frame_range_rvos
            else:
                sampling_frame_range = self.sampling_frame_range

            start_idx = max(0, ref_frame-sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            if self.multidataset and self.is_train:
                transforms = selected_augmentations[dataset_dict['dataset_source']](aug_input)
            else:
                transforms = selected_augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # if (video_annos is None) or (not self.is_train):
            #     continue
            # for SOT and VOS, we need the box anno in the 1st frame during inference
            # if (video_annos is None) or (not self.is_train):
            #     continue
            if not self.is_train:
                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # add ori_id for VOS inference
                ori_id_list = [x["ori_id"] if "ori_id" in x else None for x in annos]
                instances.ori_id = ori_id_list
                dataset_dict["instances"].append(instances)
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            # if no object in current frame, sample another frame
            instances_tmp = utils.filter_empty_instances(copy.deepcopy(instances))
            if len(instances_tmp) == 0:
                return None 
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
                instances = filter_empty_instances(instances)
            if torch.sum(instances.gt_ids != -1) == 0:
                return None
            dataset_dict["instances"].append(instances)

        # NOTE: the augmentation for RVOS must be "by_clip" if has "expression"
        if "expressions" in dataset_dict:
            # pick one expression if there are multiple expressions during training,
            dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)

        if not self.is_train:
            return dataset_dict
        # NOTE: for training VOS, only randomly select one gt instance appearing in all frames
        all_instances = dataset_dict["instances"]   # list[Instances]
        gt_ids_tmp = [instances.gt_ids.tolist() for instances in all_instances] # list[list]
        # remove the gt_id == -1, which indicates the object not in current frame
        gt_ids = []
        for gt_ids_per_frame in gt_ids_tmp:
            gt_ids.append([x for x in gt_ids_per_frame if x != -1])
        # get the intersection ids of all frames
        valid_ids = list(set(gt_ids[0]).intersection(*gt_ids[1:]))
        if len(valid_ids) == 0:
            return None
        else:
            pick_id = random.choice(valid_ids)
            new_instances = []
            # loop over all frames
            for instances in all_instances:
                # loop over all instances
                cur_ids = instances.gt_ids.tolist()
                for _ in range(len(instances)):
                    if cur_ids[_] == pick_id:
                        new_instances.append(instances[_])
                        break
            dataset_dict["instances"] = new_instances

        return dataset_dict

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')
