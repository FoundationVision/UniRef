# Training

We take the `R50` visual backbone as an example.

For the `Swin-L` visual backbone, please change the `--config-file` arguments.


## Pretraining

For the Objects365v2 pretraining, we use 32 A100 GPUs.

### Multi-node Training

On each node, run the following command. Please remember to change the `[node_rank]`, `[master_addr]`, `[master_port]` arguments.

```
python3 projects/UniRef/train_net.py \
        --num-machines=4 \
        --machine-rank=[node_rank] \
        --num-gpus=8 \
        --dist-url="tcp://[master_addr]:[master_port]" \
        --config-file projects/UniRef/configs/pretrain/obj365v2_r50_32gpu.yaml  \
        --resume \
        OUTPUT_DIR outputs/obj365v2_r50_32gpu

```

### Single-node Training

On a single node with 8 GPUs, run the following command. The training iterations are increased accordingly.

```
python3 projects/UniRef/train_net.py \
        --config-file projects/UniRef/configs/pretrain/obj365v2_r50_32gpu.yaml \
        --num-gpus 8 \
        --resume \
        OUTPUT_DIR outputs/obj365v2_r50_32gpu \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.MAX_ITER 1362884 \
        SOLVER.STEPS [1249384,] 
```


## Image-level Training

For the image-level training, we use 16 A100 GPUs. 

We find it is hard to converage when directly joint training on RIS and FSS tasks. So we separate the training process into two steps.



### Multi-node Training

On each node, run the following commands.

RIS task:

```
python3 projects/UniRef/train_net.py \
        --num-machines=2 \
        --machine-rank=[node_rank] \
        --num-gpus=8 \
        --dist-url="tcp://[master_addr]:[master_port]" \
        --config-file projects/UniRef/configs/image/joint_task_det_rec_r50_16gpu.yaml  \
        --resume \
        OUTPUT_DIR outputs/joint_task_det_rec_r50_16gpu \
        MODEL.WEIGHTS outputs/obj365v2_r50_32gpu/model_final.pth
```

RIS & FSS tasks:

```
python3 projects/UniRef/train_net.py \
        --num-machines=2 \
        --machine-rank=[node_rank] \
        --num-gpus=8 \
        --dist-url="tcp://[master_addr]:[master_port]" \
        --config-file projects/UniRef/configs/image/joint_task_finetune_det_rec_fss_r50_16gpu.yaml  \
        --resume \
        OUTPUT_DIR outputs/joint_task_det_rec_fss_r50_16gpu \
        MODEL.WEIGHTS outputs/joint_task_det_rec_r50_16gpu/model_final.pth
```

### Single-node Training

On a single node with 8 GPUs, run the following commands. The training iterations are increased accordingly.

RIS task:

```
python3 projects/UniRef/train_net.py \
        --config-file projects/UniRef/configs/image/joint_task_det_rec_r50_16gpu.yaml  \
        --num-gpus=8 \
        --resume \
        OUTPUT_DIR outputs/joint_task_det_rec_r50_16gpu \
        MODEL.WEIGHTS outputs/obj365v2_r50_32gpu/model_final.pth \
        SOLVER.MAX_ITER 180000 \
        SOLVER.STEPS [150000,] 
```

RIS & FSS tasks:

```
python3 projects/UniRef/train_net.py \
        --num-machines=2 \
        --machine-rank=[node_rank] \
        --num-gpus=8 \
        --dist-url="tcp://[master_addr]:[master_port]" \
        --config-file projects/UniRef/configs/image/joint_task_finetune_det_rec_fss_r50_16gpu.yaml  \
        --resume \
        OUTPUT_DIR outputs/joint_task_det_rec_fss_r50_16gpu \
        MODEL.WEIGHTS outputs/joint_task_det_rec_r50_16gpu/model_final.pth \
        SOLVER.MAX_ITER 60000 \
```

## Video-level Training

For the video-level training, we use 16 A100 GPUs.

### Multi-node Training

On each node, run the following command.

```
python3 projects/UniRef/train_net.py \
        --num-machines=2 \
        --machine-rank=[node_rank] \
        --num-gpus=8 \
        --dist-url="tcp://[master_addr]:[master_port]" \
        --config-file projects/UniRef/configs/video/joint_task_vos_rvos_r50_16gpu.yaml  \
        --resume \
        OUTPUT_DIR outputs/joint_task_vos_rvos_r50_16gpu \
        MODEL.WEIGHTS outputs/joint_task_det_rec_fss_r50_16gpu/model_final.pth 
```

### Single-node Training

On a single node with 8 GPUs, run the following command. The training iterations are increased accordingly.

```
python3 projects/UniRef/train_net.py \
        --config-file projects/UniRef/configs/video/joint_task_vos_rvos_r50_16gpu.yaml  \
        --num-gpus=8 \
        --resume \
        OUTPUT_DIR outputs/joint_task_vos_rvos_r50_16gpu \
        MODEL.WEIGHTS outputs/joint_task_det_rec_fss_r50_16gpu/model_final.pth \
        SOLVER.MAX_ITER 180000 \
        SOLVER.STEPS [150000,] 
```