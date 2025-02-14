# UniRef++: Segment Every Reference Object in Spatial and Temporal Spaces

Official implementation of [UniRef++](), an extended version of ICCV2023 [UniRef](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Segment_Every_Reference_Object_in_Spatial_and_Temporal_Spaces_ICCV_2023_paper.pdf).

![UniRef](assets/network.png)

## Highlights

- UniRef/UniRef++ is a unified model for four object segmentation tasks, namely referring image segmentation (RIS), few-shot segmentation (FSS), referring video object segmentation (RVOS) and video object segmentation (VOS).
- At the core of UniRef++ is the UniFusion module for injecting various reference information into network. And we implement it using flash attention with high efficiency.
- UniFusion could play as the plug-in component for foundation models like [SAM](https://github.com/facebookresearch/segment-anything).


## Schedule

- [x] Add Training Guide
- [x] Add Evaluation Guide
- [x] Add Data Preparation
- [x] Release Model Checkpoints
- [x] Release Code

## Results


https://github.com/FoundationVision/UniRef/assets/21001460/63d875ed-9f5b-47c9-998f-e83faffedbba


### Referring Image Segmentation
![RIS](assets/RIS.png)

### Referring Video Object Segmentation
![RVOS](assets/Ref-vos.png)

### Video Object Segmentation
![VOS](assets/VOS.png)

### Zero-shot Video Segmentation & Few-shot Image Segmentation
![zero-few-shot](assets/zero-few-shot.png)

## Model Zoo

#### Objects365 Pretraining


| Model             | Checkpoint |
| ------------------| :--------: |
| R50 | [model](https://drive.google.com/file/d/1cz7xWfk0xBRNMTM7P8Vtb7jT6cBv5fKN/view?usp=sharing) |
| Swin-L | [model](https://drive.google.com/file/d/1C9tjfR6puq6HUcLSwII74GRQBl-YCuD8/view?usp=sharing) |

#### Imge-joint Training

| Model             | RefCOCO | FSS-1000 | Checkpoint |
| ------------------| :----:  |  :----:  | :--------: |
| R50 | 76.3 | 85.2 | [model](https://drive.google.com/file/d/1RNerEk7nrbFBI9dY5HIK7ErmqKLN40_g/view?usp=sharing) |
| Swin-L | 79.9 | 87.7 | [model](https://drive.google.com/file/d/1dhCRuSDkw7IjxoUZo1EHDPU_608QHcx_/view?usp=sharing) |


#### Video-joint Training

The results are reported on the validation set.

  | Model             | RefCOCO | FSS-1000 | Ref-Youtube-VOS | Ref-DAVIS17 | Youtube-VOS18 | DAVIS17 | LVOS | Checkpoint |
  | ------------------| :----:  | :---: | :-----: | :---: | :--: | :--: | :-------: | :--: |
  | UniRef++-R50      |  75.6   | 79.1  |  61.5   | 63.5  | 81.9 | 81.5 |   60.1    | [model](https://drive.google.com/file/d/190SV9GU6Pd9FMZQnRrCbgw8lqDYF9_-I/view?usp=sharing) |
  | UniRef++-Swin-L   |  79.1   | 85.4  |  66.9   | 67.2  | 83.2 | 83.9 |   67.2    | [model](https://drive.google.com/file/d/1ggkoEo1n2b-3sZDVVw3qFg1kyJQPc1jT/view?usp=sharing)


## Installation

See [INSTALL.md](./INSTALL.md)

## Getting Started

Please see [DATA.md](assets/DATA.md) for data preparation.

Please see [EVAL.md](assets/EVALUATION.md) for evaluation.

Please see [TRAIN.md](assets/TRAIN.md) for training.


## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{wu2023uniref++,
  title={UniRef++: Segment Every Reference Object in Spatial and Temporal Spaces},
  author={Wu, Jiannan and Jiang, Yi and Yan, Bin and Lu, Huchuan and Yuan, Zehuan and Luo, Ping},
  journal={arXiv preprint arXiv:2312.15715},
  year={2023}
}
```

```BibTeX
@inproceedings{wu2023uniref,
  title={Segment Every Reference Object in Spatial and Temporal Spaces},
  author={Wu, Jiannan and Jiang, Yi and Yan, Bin and Lu, Huchuan and Yuan, Zehuan and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2538--2550},
  year={2023}
}
```

## Acknowledgement

The project is based on [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT) codebase. We also refer to the repositories [Detectron2](https://github.com/facebookresearch/detectron2), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [STCN](https://github.com/hkchengrex/STCN), [SAM](https://github.com/facebookresearch/segment-anything). Thanks for their awsome works!


