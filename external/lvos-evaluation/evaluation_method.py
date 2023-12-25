#!/usr/bin/env python
import argparse
import os
import sys
from ast import arg
from time import time

import numpy as np
import pandas as pd

from lvos.evaluation import LVOSEvaluation as LVOSEvaluation_SP
from lvos.evaluation_mp import LVOSEvaluation as LVOSEvaluation_MP

default_lvos_path = "../../datasets/lvos/valid/"

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--lvos_path', type=str, help='Path to the LVOS folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_lvos_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='valid')
parser.add_argument('--mp_nums', type=int, default=8, help='Multiple process numbers',)

parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',)
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)
args, _ = parser.parse_known_args()
if args.mp_nums<=1:
    args.mp_nums=1
    LVOSEvaluation=LVOSEvaluation_SP
    print(f'Evaluating with single processing.')
else:
    LVOSEvaluation=LVOSEvaluation_MP
    print(f'Evaluating with multiple processing with {args.mp_nums} processes.')




csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    if args.mp_nums<=1:
        dataset_eval = LVOSEvaluation(lvos_root=args.lvos_path, task=args.task, gt_set=args.set)
    else:
        dataset_eval = LVOSEvaluation(lvos_root=args.lvos_path, task=args.task, gt_set=args.set, mp_procs=args.mp_nums)

    metrics_res,metrics_res_seen,metrics_res_unseen = dataset_eval.evaluate(args.results_path)
    J, F ,V = metrics_res['J'], metrics_res['F'], metrics_res['V']
    J_seen, F_seen ,V_seen = metrics_res_seen['J'], metrics_res_seen['F'], metrics_res_seen['V']
    J_unseen, F_unseen ,V_unseen = metrics_res_unseen['J'], metrics_res_unseen['F'], metrics_res_unseen['V']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean','J-Mean', 'J-seen-Mean', 'J-unseen-Mean', 'F-Mean','F-seen-Mean', 'F-unseen-Mean', 'V-Mean',  'V-seen-Mean',  'V-unseen-Mean']
    #final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    final_mean = ((np.mean(J_seen["M"]) + np.mean(F_seen["M"])) + (np.mean(J_unseen["M"]) + np.mean(F_unseen["M"])))/ 4.

    g_res = np.array([final_mean, (np.mean(J_seen["M"])+np.mean(J_unseen["M"]))/2, np.mean(J_seen["M"]), np.mean(J_unseen["M"]), (np.mean(F_seen["M"])+np.mean(F_unseen["M"]))/2, np.mean(F_seen["M"]),
                      np.mean(F_unseen["M"]), (np.mean(V_seen["M"])+np.mean(V_unseen["M"]))/2, np.mean(V_seen["M"]), np.mean(V_unseen["M"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean', 'V-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    V_per_object = [V['M_per_object'][x] for x in seq_names]

    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object, V_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
sys.stdout.write('\n')