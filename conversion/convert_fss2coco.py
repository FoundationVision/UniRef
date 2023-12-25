"""
We follow the VAT (https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer) to prepare few-shot segmentation datasets.
Please organize the datasets as follows, then run our conversion files.
The
|- datasets
|-   |- splits    # train/val/test splits from VAT
|-   |- fss-1000
|-   |-   |- images
|-   |-   |-    |- ab_wheel
|-   |-   |-    |- ...
"""

import json
import argparse
import os
from PIL import Image
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from torch._C import import_ir_module_from_buffer
from detectron2.structures import PolygonMasks
import pycocotools.mask as mask_util
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("image to coco annotation format.")
    parser.add_argument("--src_path", default="datasets/", type=str, help="")
    parser.add_argument("--mask_format", default="rle", choices=["polygon", "rle"], type=str)
    return parser.parse_args()


def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def mask2polygon(input_mask):
    contours, hierarchy = cv2.findContours(input_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        result.append(segmentation)
    return result

def mask2rle(input_mask):
    rle = mask_util.encode(np.array(input_mask, order="F", dtype="uint8"))
    if not isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if __name__ == "__main__":
    args = parse_args()

    dataset_name = 'fss-1000'
    splits = ["train", "val", "test"]
    for split in splits:
        # read split files
        if split == "train":
            split_name = "trn"
        else:
            split_name = split
        split_path = os.path.join(args.src_path, 'splits/fss', split_name + '.txt')
        with open(split_path, 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
        
        # read images and gt masks
        dataset_path = os.path.join(args.src_path, dataset_name)
        images, masks = [], []
        support_image_names = {}
        for c in classes:
            assert os.path.exists(os.path.join(dataset_path, 'images', c))
            imgs = os.listdir(os.path.join(dataset_path, 'images', c)) 
            img_list = [img for img in imgs if 'jpg' in img[-3:].lower()]
            img_list = sorted([os.path.join(c, img) for img in img_list])
            msk_list = sorted([img.replace('.jpg', '.png') for img in img_list])
            # add support images
            for img in img_list:
                support_image_names[img] = sorted([support_img for support_img in img_list if support_img != img])
            images.extend(img_list)
            masks.extend(msk_list)
        # images and masks correspond as 1-to-1, as there is only one gt mask in each image
        num_images = len(images)
        print(f"{dataset_name} {split} split has {num_images} images.") # 520/240/240 in train/val/test, each class has 10 images

        # create {img_name: img_id}
        img2id = {}
        for i, img in enumerate(images):
            img2id[img] = i + 1  # start from 1
        
        # create anno path 
        anno_path = os.path.join(dataset_path, "annotations")
        os.makedirs(anno_path, exist_ok=True)
        # "support_image_names" and "support_image_ids" will be saved in "images"
        des_dataset = {"images": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}], "annotations": []}
        img_idx, ann_idx = 0, 0
        for idx in tqdm(range(num_images)):
            image = images[idx]
            mask = masks[idx]
            assert image[:-4] == mask[:-4]
            image_path = os.path.join(dataset_path, 'images', image)
            mask_path  = os.path.join(dataset_path, 'images', mask)
            H, W, _ = cv2.imread(image_path).shape
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_h, mask_w = mask.shape
            mask_h, mask_w = mask.shape
            if mask_h != H or mask_w != W:
                print(f"{image_path} image and mask sizes are incompatible!")
                continue
            mask_cur = mask / 255
            mask_cur = (mask_cur > 0.5).astype(np.uint8)    # 0, 1 binary
            if not (mask_cur > 0).any():
                print(f"{image_path} does not have available mask!")
                continue

            # append "images"
            img_idx += 1
            img_dict ={}
            img_dict["file_name"] = image
            img_dict["height"], img_dict["width"] = H, W
            img_dict["id"] = img_idx
            # support images
            support_images = support_image_names[image] # list[dict]
            support_image_ids = [img2id[img] for img in support_images]
            img_dict["support_image_names"] = support_images
            img_dict["support_image_ids"] = support_image_ids
            img_dict["split"] = split
            des_dataset["images"].append(img_dict)

            # append "annotations"
            ann_idx += 1
            ann_dict = {}
            ann_dict["image_id"], ann_dict["id"], ann_dict["iscrowd"], ann_dict["category_id"] = \
                img_idx, ann_idx, 0, 1
            box = bounding_box(mask_cur)
            area = int(box[-2] * box[-1])
            ann_dict["bbox"] = box
            ann_dict["area"] = area
            if args.mask_format == "polygon":
                ann_dict["segmentation"] = mask2polygon(mask_cur)
            elif args.mask_format == "rle":
                ann_dict["segmentation"] = mask2rle(mask_cur)
            else:
                raise NotImplementedError
            des_dataset["annotations"].append(ann_dict)
        
        # save
        output_json = os.path.join(anno_path, f"{split}.json")
        json.dump(des_dataset, open(output_json, 'w'))

            
