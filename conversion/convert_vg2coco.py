import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/coco2014/annotations/pretraining-vg/instances.json", type=str, help="")
    parser.add_argument("--des_json", default="datasets/visual_genome/instances_vg.json", type=str, help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src_dataset = json.load(open(args.src_json, 'r'))["train"]
    src_vg = []
    for anno_dict in src_dataset:
        if anno_dict["data_source"] == "vg":
            src_vg.append(anno_dict)

    des_dataset = {'images':[], 'categories':[{"supercategory": "object","id": 1,"name": "object"}], 'annotations':[]}

    # append videos and annotations
    anno_id = 0
    for anno_dict in src_vg:
        anno_id += 1
        # images
        img_dict = {}
        img_dict["file_name"] = str(anno_dict["image_id"]) + ".jpg"
        img_dict["width"], img_dict["height"], img_dict["id"] = anno_dict["width"], anno_dict["height"], anno_id
        img_dict["expressions"] = anno_dict["expressions"]
        des_dataset["images"].append(img_dict)

        # annotations
        anno_dict_new = {}
        anno_dict_new["iscrowd"], anno_dict_new["category_id"], anno_dict_new["id"] = \
            0, 1, anno_id
        anno_dict_new["image_id"] = anno_id
        anno_dict_new["bbox"] = anno_dict["bbox"]  # x1, y1, w, h
        anno_dict_new["areas"] = anno_dict["bbox"][-2] * anno_dict["bbox"][-1]
        des_dataset["annotations"].append(anno_dict_new)
    # save
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)
