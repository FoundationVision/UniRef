# LVOS Semi-supervised evaluation package

This package is used to evaluate semi-supervised long-term video multi-object segmentation models for the <a href="https://lingyihongfd.github.io/lvos.github.io/" target="_blank">LVOS</a> dataset.

This tool is also used to evaluate the submissions in the Codalab site for the <a href="https://codalab.lisn.upsaclay.fr/competitions/8767" target="_blank">Semi-supervised LVOS Challenge</a>.

### Installation

```bash
# Download the code
git clone https://github.com/LingyiHongfd/lvos-evaluation.git && cd lvos-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```

If you don't want to specify the LVOS path every time, you can modify the default value in the variable `default_lvos_path` in `evaluation_method.py`(the following examples assume that you have set it).

Note: `default_lvos_path` is the valid split path.

Otherwise, you can specify the path in every call by using the flag `--lvos_path /path/to/LVOS` when calling `evaluation_method.py`.

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results:

- `global_results.csv` contains the overall results.
- `per-sequence_results.csv` contain the per sequence.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Evaluate LVOS Semi-supervised

In order to evaluate your semi-supervised method in LVOS, execute the following command substituting `results/semi-supervised/ddmemory` by the folder path that contains your results:

```bash
python evaluation_method.py --task semi-supervised --results_path results/semi-supervised/ddmemory --mp_nums 1
```

The semi-supervised results have been generated using DDMemory.

For some reason, the result of DDMemory is unavailable temporarily. So we provide the result of <a href="https://github.com/yoxu515/aot-benchmark" target="_blank"> AOT-T </a> as an alternative. You can download the result <a href="https://drive.google.com/drive/folders/1bGbyNUdbvmQBBezVv_3Fp-5LITMsY2EG?usp=share_link" target="_blank"> here </a> and unzip the file. After putting the unziped file under the folder `results/semi-supervised/aott`, please use the following command to evaluate AOT-T result.

```bash
python evaluation_method.py --task semi-supervised --results_path results/semi-supervised/aott --mp_nums 1
```

`mp_nums` is set as 1 by default. Because the score computing process in serial mode is time-consuming, you can set `mp_nums` larger than 1 (such as 2) to enable multiple processing and speed up the evaluation. But we suggest that `mp_nums` should be set to less than 8 on a regular server.   

## Acknowledgement

The codes are modified from <a href="https://github.com/davisvideochallenge/davis2017-evaluation"> DAVIS 2017 Semi-supervised and Unsupervised evaluation package</a>.

## Citation

Please cite both papers in your publications if LVOS or this code helps your research.

```latex
@article{hong2022lvos,
    title={LVOS: A Benchmark for Long-term Video Object Segmentation},
    author={Hong, Lingyi and Chen, Wenchao and Liu, Zhongying and Zhang, Wei and Guo, Pinxue and Chen, Zhaoyu and Zhang, Wenqiang},
    journal={arXiv preprint arXiv:2211.10181},
    year={2022},
}
```
