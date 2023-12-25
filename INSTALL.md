## Install：

## 目前仅支持本地安装：

```
export http_proxy=10.20.47.147:3128 https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
pip3 install -e . --user
pip3 install --user shapely==1.7.1
pip3 install git+https://github.com/XD7479/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI" --user
```

（国内机器使用“export...” 代理加速）

## 链接数据集地址：

```
ln -s /path_to_coco_dataset datasets/coco
ln -s /path_to_ytvis2019_dataset datasets/ytvis_2019
ln -s /path_to_ytvis2021_dataset datasets/ytvis_2021
```



## 数据集HDFS 地址（hl）：

COCO：hdfs://haruna/home/byte_arnold_hl_vc/user/wujunfeng/data/coco

YTVIS-2019：hdfs://haruna/home/byte_arnold_hl_vc/user/wujunfeng/data/ytvis.tar

YTVIS-2021：hdfs://haruna/home/byte_arnold_hl_vc/user/wujunfeng/data/ytvis21.tar

YTVIS-2022val：hdfs://haruna/home/byte_arnold_hl_vc/user/wujunfeng/data/valid_ytvis2022.zip

OVIS：hdfs://haruna/home/byte_arnold_hl_vc/user/wujunfeng/data/OVIS.tar.gz



美东地址：

hdfs://harunava/home/byte_ad_audit/bi_algorithm/junfeng.wu/data/



## Deformable Transformer build:

DDETRS和IDOL都使用了DeformableTransformer， 需要进入projects/XXX/XXX/model/ops 编译

```
sh mask.sh
```

