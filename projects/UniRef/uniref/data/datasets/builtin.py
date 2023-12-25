# -*- coding: utf-8 -*-

from atexit import register
from multiprocessing.sharedctypes import Value
import os
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .refcoco import (
    register_refcoco,
    _get_refcoco_meta,
)
from .fss import (
    register_fss,
    _get_fss_meta,
)

from .ytvis import register_ytvis_instances


# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO = {
    # refcoco
    "refcoco-unc-train": ("coco2014/train2014", "coco2014/annotations/refcoco-unc/instances_train.json"),
    "refcoco-unc-val": ("coco2014/train2014", "coco2014/annotations/refcoco-unc/instances_val.json"),
    "refcoco-unc-testA": ("coco2014/train2014", "coco2014/annotations/refcoco-unc/instances_testA.json"),
    "refcoco-unc-testB": ("coco2014/train2014", "coco2014/annotations/refcoco-unc/instances_testB.json"),
    # refcocog
    "refcocog-umd-train": ("coco2014/train2014", "coco2014/annotations/refcocog-umd/instances_train.json"),
    "refcocog-umd-val": ("coco2014/train2014", "coco2014/annotations/refcocog-umd/instances_val.json"),
    "refcocog-umd-test": ("coco2014/train2014", "coco2014/annotations/refcocog-umd/instances_test.json"),
    # refcoco+
    "refcocoplus-unc-train": ("coco2014/train2014", "coco2014/annotations/refcocoplus-unc/instances_train.json"),
    "refcocoplus-unc-val": ("coco2014/train2014", "coco2014/annotations/refcocoplus-unc/instances_val.json"),
    "refcocoplus-unc-testA": ("coco2014/train2014", "coco2014/annotations/refcocoplus-unc/instances_testA.json"),
    "refcocoplus-unc-testB": ("coco2014/train2014", "coco2014/annotations/refcocoplus-unc/instances_testB.json"),
    # mixed
    "refcoco-mixed": ("coco2014/train2014", "coco2014/annotations/refcoco-mixed/instances_train.json"),
    # vg, for pretraining
    "visual_genome": ("visual_genome/images", "visual_genome/annotations/instances_vg.json"),
}


def register_all_refcoco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFCOCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        if "visual_genome" in key:
            dataset_name_in_dict = "visual_genome"
        elif "refcoco" in key:
            dataset_name_in_dict = "refcoco"
        else:
            raise ValueError
        register_refcoco(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict=dataset_name_in_dict
        )

# ==== Predefined splits for FSS datasets ===========
_PREDEFINED_SPLITS_FSS = {
    # fss-1000
    "fss-1000-train": ("fss-1000/images", "fss-1000/annotations/train.json"),
    "fss-1000-val": ("fss-1000/images", "fss-1000/annotations/val.json"),
    "fss-1000-test": ("fss-1000/images", "fss-1000/annotations/test.json"),
}

def register_all_fss(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_FSS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_fss(
            key,
            _get_fss_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for Video-COCO datasets ===========
# coco pseudo videos
def _get_coco_video_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

_PREDEFINED_SPLITS_COCO_VIDS = {
    "video-coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017_video.json"),
    "video-coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017_video.json"),
}

def register_all_coco_videos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_coco_video_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ======================================================================
# vos /refvos datasets
VOS_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}] # only one class for VOS

def _get_vos_meta():
    thing_ids = [k["id"] for k in VOS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# ==== Predefined splits for YTVOS datasets ===========
_PREDEFINED_SPLITS_YTVOS = {
    "ytbvos18-train": ("ytbvos18/train/JPEGImages", "ytbvos18/annotations/train.json"),
    "ytbvos18-val": ("ytbvos18/valid/JPEGImages", "ytbvos18/annotations/valid.json"),
    "ytbvos19-train": ("ytbvos19/train/JPEGImages", "ytbvos19/annotations/train.json"),
    "ytbvos19-val": ("ytbvos19/valid/JPEGImages", "ytbvos19/annotations/valid.json"),
    "davis17-train": ("davis17/DAVIS/JPEGImages/480p", "davis17/annotations/davis2017_train.json"),
    "davis17-val": ("davis17/DAVIS/JPEGImages/480p", "davis17/annotations/davis2017_val.json"),
    "ovis-train": ("ovis/train", "ovis/annotations/train.json"),
    "mose-train": ("mose/train/JPEGImages", "mose/annotations/train.json"),
    "mose-val": ("mose/valid/JPEGImages", "mose/annotations/valid.json"),
    "vos-lvos-train": ("lvos/train/JPEGImages", "lvos/annotations_vos/train.json"),
    "vos-lvos-val": ("lvos/valid/JPEGImages", "lvos/annotations_vos/valid.json"),
}

def register_all_ytvos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_vos_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for REFYTVOS datasets ===========
_PREDEFINED_SPLITS_REFYTVOS = {
    "refytvos-train": ("ref-youtube-vos/train/JPEGImages", "ref-youtube-vos/annotations/train.json"),
    "refytvos-vl-train": ("ref-youtube-vos/train/JPEGImages", "ref-youtube-vos/annotations/train.json"),
    "refytvos-val": ("ref-youtube-vos/valid/JPEGImages", "ref-youtube-vos/annotations/valid.json"),
    "refdavis-val-0": ("ref-davis/DAVIS/JPEGImages/480p", "ref-davis/annotations/valid_0.json"),
    "refdavis-val-1": ("ref-davis/DAVIS/JPEGImages/480p", "ref-davis/annotations/valid_1.json"),
    "refdavis-val-2": ("ref-davis/DAVIS/JPEGImages/480p", "ref-davis/annotations/valid_2.json"),
    "refdavis-val-3": ("ref-davis/DAVIS/JPEGImages/480p", "ref-davis/annotations/valid_3.json"),
    "video-refcoco-mixed": ("coco2014/train2014", "coco2014/annotations/refcoco-mixed/instances_train_video.json"),
}


def register_all_refytvos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFYTVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_vos_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_refcoco(_root)
    register_all_fss(_root)
    register_all_coco_videos(_root)
    register_all_ytvos(_root)
    register_all_refytvos(_root)
