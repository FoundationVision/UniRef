# Installation 

## Requirements

we run our code in the following environment:

- CUDA 11.7
- Python 3.9
- Pytorch 2.1.0

## Setup

First, clone the repository.

```
git clone https://github.com/FoundationVision/UniRef.git
```

Second, install the detectron2.

```
pip3 install -e . --user
```

Third, install the necessary packages.

```
pip3 install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI" --user
pip3 install -r requirements.txt --user
# flash-attn
python3 -m pip install ninja
python3 -m pip install flash-attn==2.0.6
```

Finally, compile deformable attention CUDA operator.

```
cd projects/UniRef/uniref/models/deformable_detr/ops;
bash make.sh
```



