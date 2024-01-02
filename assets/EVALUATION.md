# Evaluation

We take the checkpoint `video-joint_r50.pth` as an example. We have listed all the evaluation datasets in the [config files](https://github.com/FoundationVision/UniRef/tree/main/projects/UniRef/configs/eval).

For the swin-L backbone, please change the  `--config-file` and `MODEL.WEIGHTS` arguments.

## RIS

```
python3 projects/UniRef/train_net.py --config-file projects/UniRef/configs/eval/r50/eval_rec_r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS video-joint_r50.pth 
```

## FSS

1-shot evaluation.

```
python3 projects/UniRef/train_net.py --config-file projects/UniRef/configs/eval/r50/eval_fss_r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS video-joint_r50.pth
```

5-shot evaluation.

```
python3 projects/UniRef/train_net.py --config-file projects/UniRef/configs/eval/r50/eval_fss_r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS video-joint_r50.pth TASK.FSS.NSHOT 5
```

## RVOS

```
python3 projects/UniRef/train_net.py --config-file projects/UniRef/configs/eval/r50/eval_rvos_r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS video-joint_r50.pth
```

- Ref-Youtube-VOS

Run the following commands. Then, submit the `submission.zip` to [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).

```
cd /path/to/output
mv refytvos Annotations
zip -q -r submission.zip Annotations
```



- Ref-DAVIS

Run the following commands. The final results are the average metric scores across the 4 splits.

```
cd external/davis2017-evaluation/
python3 evaluation_method.py --results_path /path/to/output/refdavis/refdavis-val-0
python3 evaluation_method.py --results_path /path/to/output/refdavis/refdavis-val-1
python3 evaluation_method.py --results_path /path/to/output/refdavis/refdavis-val-2
python3 evaluation_method.py --results_path /path/to/output/refdavis/refdavis-val-3
```


## VOS

```
python3 projects/UniRef/train_net.py --config-file projects/UniRef/configs/eval/r50/eval_vos_r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS video-joint_r50.pth
```

- Youtube-VOS-18

Run the following commands. Then, submit the `submission.zip` to [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7685#participate-submit_results).

```
cd /path/to/output
mv ytbvos18 Annotations
zip -q -r submission.zip Annotations
```

- Youtube-VOS-19

Run the following commands. Then, submit the `submission.zip` to [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/6066#participate-submit_results).

```
cd /path/to/output
mv ytbvos19 Annotations
zip -q -r submission.zip Annotations
```

- DAVIS17

```
cd external/davis2017-evaluation
python3 evaluation_method.py --task semi-supervised --results_path /path/to/output/davis17
```

- LVOS

```
cd external/lvos-evaluation
python3 evaluation_method.py --results_path /path/to/output/lvos-vos
```

- MOSE

Run the following commands. Then, submit the `submission.zip` to [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/10703#participate-submit_results).

```
cd /path/to/output/mose
zip -q -r submission.zip *
```