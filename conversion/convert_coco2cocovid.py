import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/coco/annotations/instances_val2017.json", type=str, help="")
    parser.add_argument("--des_json", default="datasets/coco/annotations/instances_val2017_video.json", type=str, help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src_path = os.path.dirname(args.src_json)
    des_path = os.path.dirname(args.des_json)

    splits = ['train', 'val']
    for split in splits:
        print(f'Converting COCO {split} split to video annotations...')
        src_json = os.path.join(src_path, f'instances_{split}2017.json')
        src_dataset = json.load(open(src_json, 'r')) 
        des_dataset = {'videos':[], 'categories':[], 'annotations':[]}
        des_dataset["categories"] = src_dataset["categories"]
        # videos
        for img_dict in src_dataset["images"]:
            vid_dict = {}
            vid_dict["length"] = 1
            vid_dict["file_names"] = [img_dict["file_name"]]
            vid_dict["width"], vid_dict["height"], vid_dict["id"] = img_dict["width"], img_dict["height"], img_dict["id"]
            des_dataset["videos"].append(vid_dict)
        # annotations
        for anno_dict in src_dataset["annotations"]:
            anno_dict_new = {}
            anno_dict_new["iscrowd"], anno_dict_new["category_id"], anno_dict_new["id"] = \
                anno_dict["iscrowd"], anno_dict["category_id"], anno_dict["id"]
            anno_dict_new["video_id"] = anno_dict["image_id"]
            anno_dict_new["bboxes"] = [anno_dict["bbox"]]
            if "segmentation" in anno_dict:
                anno_dict_new["segmentations"] = [anno_dict["segmentation"]]
            anno_dict_new["areas"] = [anno_dict["area"]]
            des_dataset["annotations"].append(anno_dict_new)
        # save
        des_json = os.path.join(des_path, f'instances_{split}2017_video.json')
        with open(des_json, "w") as f:
            json.dump(des_dataset, f)