# Data Preparation

## Pretrained Weights

The pretrained weights are placed in the folder `pretrained_models`.

- Visual Backbones

    -  R-50: please download from [Detectron2](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl) or [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EQDlciMpUA9OnPUyv5Kj9PUBlFn2tIjec25uo2eYySgePQ?e=a9Z8sj).
    -  Swin-L: please download from [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EXqgJ6QUfiRGlz9cMckkDkgB9BBs41rb12hSK5Gxa3w0lQ?e=wErbvY), which is converted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).


- Text Encoders

    -    BERT-base: please download from [Hugging Face](https://huggingface.co/bert-base-uncased).


- SAM
    - SAM-H: please download form [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

After preparation, the folder structure should be like:

```
|- datasets/
|- detectron2/
|- projects/
|    |- Uniref/
|- pretrained_models/
|    |- R-50.pkl
|    |- swin_large_patch4_window12_384_22k.pkl
|    |- sam_vit_h_4b8939.pth
|    |- bert-base-uncased/
...
```


## DATA

We list the data for training and inference as following. The datasets in brackets `()` are only used for inference.

- **Pretraining**: 
    - Objects365
- **Image-level Training**
    - DET: COCO2017
    - RIS: RefCOCO/+/g
    - FSS: FSS-1000
- **Video-level Training**
    - RVOS: RefCOCO/+/g, Ref-Youtube-VOS, (Ref-DAVIS17)
    - VOS: COCO2017, Youtube-VOS-19, LVOS, OVIS, (Youtube-VOS-18, DAVIS17, MOSE)

We mainly follow [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT/blob/master/assets/DATA.md) to prepare our data. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wjn922_connect_hku_hk/Euz3hhalJEVEoxoLLEV0UCkBDtXD9qm2xTb-4rkCxMEjgQ?e=inBpdM). If you are interested in the preprocessing, please see our [conversion files](https://github.com/FoundationVision/UniRef/tree/main/conversion).

The datasets are placed in the folder `datasets`. 


### Pretraining

We provide the conversion file for downloading Objects365v2 images.

```
python3 conversion/download_objects365_v2.py
```

We use the same preprocessed json file as UNINEXT in [OneDrive](https://maildluteducn-my.sharepoint.com/personal/yan_bin_mail_dlut_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyan%5Fbin%5Fmail%5Fdlut%5Fedu%5Fcn%2FDocuments%2Foutputs%2Fzhiyuan%5Fjson%2Ezip&parent=%2Fpersonal%2Fyan%5Fbin%5Fmail%5Fdlut%5Fedu%5Fcn%2FDocuments%2Foutputs&ga=1). The data structure should be like:

```
|- datasets/
|    |- Objects365V2/
|    |    |- annotations/
|    |    |    |- zhiyuan_objv2_train_new.json
|    |    |    |- zhiyuan_objv2_val_new.json
|    |    |- images/
```


### Image-level Training


- COCO

Please download [COCO2017](https://cocodataset.org/#home) from official website. The annotation file for video-level training is provided in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ER8ip0Znxv1Gk595OU8CsvABC3ti8nGdpNfbnHy8sEcpgg?e=pC6ooU). The data structure should be like:

```
|- datasets/
|    |- coco/
|    |    |- annotations/
|    |    |    |- instances_train2017_video.json
|    |    |    |- instances_train2017.json
|    |    |    |- instances_val2017.json
|    |    |- train2017/
|    |    |- val2017/
```

- RefCOCO/+/g

Please download [COCO2014](https://cocodataset.org/#home) images from official website. The original annotation files are from [SeqTR](https://github.com/seanzhuh/SeqTR). We further convert the files and provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EYBg1bkrn5VEuSzaFiO3-OkB4yIK4M3xDZfw2f9WpGokmg?e=0YdXOi). The data structure should be like:

```
|- datasets/
|    |- coco2014/
|    |    |- annotations/
|    |    |    |- refcoco-mixed/
|    |    |    |- refcoco-unc/
|    |    |    |- refcocoplus-unc/
|    |    |    |- refcocog-umd/
|    |    |- train2014/
```


- FSS-1000

Please download [FSS-1000](https://github.com/HKUSTCV/FSS-1000) from official repo. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Ed1UUcBHVv1Ngn6aa8w29ccBfFcv8UUx3lE4XU1DQcrdkQ?e=8FGzeE). The data structure should be like:

```
|- datasets/
|    |- fss-1000/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |    |- test.json
|    |    |- images/
```



### Video-level Training

- Ref-Youtube-VOS

Please download [Ref-Youtube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-get-data) from official website. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EaPCFzAQG7xMqLTpIp3C8y4BSyds0QvzYUUHMC5u4Q1urA?e=xrVQSs). The data structure should be like:


```
|- datasets/
|    |- ref-youtube-vos/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |- train/
|    |    |    |- JPEGImages/
|    |    |- valid/
|    |    |    |- JPEGImages/
```


- Ref-DAVIS17

Please download [Ref-DAVIS17](https://davischallenge.org/davis2017/code.html) from official website. You only need to download `DAVIS-2017-Unsupervised-trainval-480p.zip` and unzip it. You can also download the original text annotations from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions). We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EXplb3xay51HvsYkXkUqnpsBhC-Gfsu6U4zHqJzunZ9OWg?e=rDa7kX). The data structure should be like:


```
|- datasets/
|    |- ref-davis/
|    |    |- annotations/
|    |    |    |- valid_0.json
|    |    |    |- valid_1.json
|    |    |    |- valid_2.json
|    |    |    |- valid_3.json
|    |    |- DAVIS/
|    |    |    |- JPEGImages/
```


- Youtube-VOS-18

Please download [Youtube-VOS-18](https://codalab.lisn.upsaclay.fr/competitions/7685#participate) from official website. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ES7642WZKIxLoNsz83_Gj3YBmqd1Rn3VOnVjSj5IVsOOtg?e=OVAgnO). The data structure should be like:


```
|- datasets/
|    |- ytbvos18/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |- train/
|    |    |    |- JPEGImages/
|    |    |- valid/
|    |    |    |- JPEGImages/
```


- Youtube-VOS-19

Please download [Youtube-VOS-19](https://codalab.lisn.upsaclay.fr/competitions/6066#participate) from official website. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ET3BNi1Mn7RFh-U1ETyj6bwBqZt2bWqHi8Uskp_U0ZYKKQ?e=QvyDvS). The data structure should be like:


```
|- datasets/
|    |- ytbvos19/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |- train/
|    |    |    |- JPEGImages/
|    |    |- valid/
|    |    |    |- JPEGImages/
```


- DAVIS17

Please download [DAVIS17](https://davischallenge.org/davis2017/code.html) from official website. You only need to download `DAVIS-2017-trainval-480p.zip` and unzip it. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESwVnJSkbvFBmw3BwQr4NLUB3cFo0GtuV-k6R_pD0qxLDA?e=8cq4Wh). The data structure should be like:

```
|- datasets/
|    |- davis17/
|    |    |- annotations/
|    |    |    |- davis2017_train.json
|    |    |    |- davis2017_val.json
|    |    |- DAVIS/
|    |    |    |- JPEGImages/
```


- OVIS

Please download [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate) from official website. This is an video instance segmentation dataset, we convert the annotation file to class-agnostic format for our training. The preprocessed annotation file is provided in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EdbfYYhhOf1MmNLqQYXXjjUBzZTHzXxFvZieiR8AYVZ3sA?e=F45tOQ). The data structure should be like:


```
|- datasets/
|    |- ovis/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |- train/
```


- LVOS

Please download [LVOS](https://lingyihongfd.github.io/lvos.github.io/dataset.html) from official website. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ERyyEjDDqJ5GlMJQxDdbdngBzakHfXoFtizf8BD9xacpbQ?e=GohQJ4). The data structure should be like:

```
|- datasets/
|    |- lvos/
|    |    |- annotations_vos/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |- train/
|    |    |    |- JPEGImages/
|    |    |- valid/
|    |    |    |- JPEGImages/
```


- MOSE

Please download [MOSE](https://codalab.lisn.upsaclay.fr/competitions/10703#participate-get_data) from official website. We provide the preprocessed annotation files in [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EV1QQIY71G1Ll6hV2GjsCBABy4YzHYh8Jqq-f-iWK32ynA?e=5Mp2Xn). The data structure should be like:

```
|- datasets/
|    |- mose/
|    |    |- annotations/
|    |    |    |- train.json
|    |    |    |- val.json
|    |    |- train/
|    |    |    |- JPEGImages/
|    |    |- valid/
|    |    |    |- JPEGImages/
```