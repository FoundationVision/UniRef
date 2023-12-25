# 项目介绍

该项目基于Detectron2, 关注于Image level的Detection、Instance Segmentation，以及集成各种科学涨点的方法，并持续帮助其他video相关的下游任务如MOT、MOTS、VIS、RefVIS等等。所有项目在projects里，可以插拔且替换成最新的Detectron2 版本

## 目前支持的特征
04.12 支持使用RAFT进行上采样，提升Mask的预测精度

04.13 支持Large-scale Jittering数据增强以及多机训练

04.14 支持一键运行(下载数据，安装依赖，模型训练)，支持elastic训练（自动resume）

## 注意
（1） 下面这两个文件是截然不同的，用的时候一定要注意区分
https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl

https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

（2）resume的时候，必须每个节点都下载最新的权重！如果只有节点0下载了最新权重，则训练结果是不正确的

（3）D2默认支持的数据集在detectron2/data/datasets/builtin.py文件里

（4）在LVISv0.5上评测：
    COCOfy the original json: python3 datasets/prepare_cocofied_lvis.py
 (5) 如果需要使用多张图片的数据增强，可以修改detectron2/data/common.py下面的MapDataset这个类

 (6) Objects365有两代，第一代训练集约60W张图，第二代训练集有大约170W张图，DINO用的是第二代，我们这里和他保持一致。下载好原始数据后，需要用transform_obj365v2_json.py对json做一些处理，包括对图片路径重命名，重新划分训练集和测试集，以及清洗数据（训练集json中有3张图不存在）。除此之外，这两代Objects365的类别顺序发生了变化，在和COCO的80类进行转换时需要注意

## 目前包含的任务

1. DDETRS： DeformableDETR + CondInst + OTA [https://bytedance.feishu.cn/docx/doxcnrhGyKuzbnhNfqzhhrIO01c]
2. IDOL：Contrastive Online VIS model [https://bytedance.feishu.cn/docx/doxcn7iQxtqF1MdjXr5Y3Cyxwjg]
3. SeqFormer：WIP


## Usage

准备数据：
```
bash prepare_data.sh
```

### DDETRS

训练：

```
单机
python3 launch.py --nn 1 --prepare_data 0 --config-file projects/DDETRS/configs/ddetrs_baseline.yaml --resume OUTPUT_DIR ./DDETRS_BASELINE HDFS_DIR hdfs://haruna/home/byte_arnold_hl_vc/user/yanbin.iiau/UnicorNext/DDETRS_BASELINE

多机
python3 launch.py --prepare_data 0 --config-file projects/DDETRS/configs/ddetrs_baseline_16g.yaml --resume OUTPUT_DIR ./DDETRS_BASELINE_16G HDFS_DIR hdfs://haruna/home/byte_arnold_hl_vc/user/yanbin.iiau/UnicorNext/DDETRS_BASELINE_16G
```

测试

```bash
#eval
python3 projects/DDETRS/train_net.py  --num-gpus 8 --config-file projects/DDETRS/configs/base_coco.yaml --eval-only MODEL.OTA True/False MODEL.WEIGHTS xxxxx.pth
```

## DN-DETR

训练：

```
# convnext-large coco
python3 launch_dn.py --mode distribute --prepare_data 0 --backbone convnext_large_384 --exp_name convnext_large_ep50 --epochs 50 --lr_drop 40 --dataset_file coco # 16 GPUs
# convnext-large obj365 pretraining
python3 launch_dn.py --mode distribute --prepare_data 0 --use_obj365_only 1 --backbone convnext_large_384 --exp_name convnext_large_obj365_pretrain --epochs 25 --lr_drop 20 --dataset_file obj365_v2 # 32 GPUs

```