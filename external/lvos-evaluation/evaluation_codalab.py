#!/usr/bin/env python
from ast import arg
import os
import sys
from time import time
import argparse

import numpy as np
import pandas
from lvos.evaluation_mp import LVOSEvaluation as LVOSEvaluation_MP
from lvos.evaluation import LVOSEvaluation as LVOSEvaluation_SP

task = 'semi-supervised'
gt_set = 'test'

time_start = time()
# as per the metadata file, input and output directories are the arguments
if len(sys.argv) < 3:
    input_dir = "input_dir"
    output_dir = "output_dir"
    debug = True
else:
    [_, input_dir, output_dir] = sys.argv
    debug = False

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
submission_path = os.path.join(input_dir, 'res','Annotations')
if not os.path.exists(submission_path):
    sys.exit('Could not find submission file {0}'.format(submission_path))

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
gt_path = os.path.join(input_dir, 'ref')
if not os.path.exists(gt_path):
    sys.exit('Could not find GT file {0}'.format(gt_path))


# Create dataset
#dataset_eval = LVOSEvaluation_SP(lvos_root=gt_path, gt_set=gt_set, task=task, codalab=True)
dataset_eval = LVOSEvaluation_MP(lvos_root=gt_path, gt_set=gt_set, task=task, codalab=True, mp_procs=4)

# Check directory structure
res_subfolders = os.listdir(submission_path)
sys.stdout.write(submission_path)
if len(res_subfolders) == 1:
    sys.stdout.write(
        "Incorrect folder structure, the folders of the sequences have to be placed directly inside the "
        "zip.\nInside every folder of the sequences there must be an indexed PNG file for every frame.\n"
        "The indexes have to match with the initial frame.\n")
    sys.exit()

# Check that all sequences are there
missing = False
for seq in dataset_eval.dataset.get_sequences():
    if seq not in res_subfolders:
        sys.stdout.write(seq + " sequence is missing.\n")
        missing = True
if missing:
    sys.stdout.write(
        "Verify also the folder structure, the folders of the sequences have to be placed directly inside "
        "the zip.\nInside every folder of the sequences there must be an indexed PNG file for every frame.\n"
        "The indexes have to match with the initial frame.\n")
    sys.exit()

metrics_res,metrics_res_seen,metrics_res_unseen = dataset_eval.evaluate(submission_path, debug=debug)
J, F ,V = metrics_res['J'], metrics_res['F'], metrics_res['V']
J_seen, F_seen ,V_seen = metrics_res_seen['J'], metrics_res_seen['F'], metrics_res_seen['V']
J_unseen, F_unseen ,V_unseen = metrics_res_unseen['J'], metrics_res_unseen['F'], metrics_res_unseen['V']

# Generate output to the stdout
seq_names = list(J['M_per_object'].keys())
if gt_set == "val" or gt_set == "train" or gt_set == "test":
    sys.stdout.write("----------------Global results in CSV---------------\n")
    g_measures = ['Overall','J-Mean', 'J-seen-Mean', 'J-unseen-Mean', 'F-Mean','F-seen-Mean', 'F-unseen-Mean', 'V-Mean',  'V-seen-Mean',  'V-unseen-Mean']    
    final_mean = ((np.mean(J_seen["M"]) + np.mean(F_seen["M"])) + (np.mean(J_unseen["M"]) + np.mean(F_unseen["M"])))/ 4.

    g_res = np.array([final_mean, (np.mean(J_seen["M"])+np.mean(J_unseen["M"]))/2, np.mean(J_seen["M"]), np.mean(J_unseen["M"]), (np.mean(F_seen["M"])+np.mean(F_unseen["M"]))/2, np.mean(F_seen["M"]),
                      np.mean(F_unseen["M"]), (np.mean(V_seen["M"])+np.mean(V_unseen["M"]))/2, np.mean(V_seen["M"]), np.mean(V_unseen["M"])])
    table_g = pandas.DataFrame(data=np.reshape(g_res, [1, len(g_res)]), columns=g_measures)
    table_g.to_csv(sys.stdout, index=False, float_format="%0.3f")

    sys.stdout.write("\n\n------------Per sequence results in CSV-------------\n")
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean', 'V-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    V_per_object = [V['M_per_object'][x] for x in seq_names]
    table_seq = pandas.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object, V_per_object)), columns=seq_measures)
    table_seq.to_csv(sys.stdout, index=False, float_format="%0.3f")

# Write scores to a file named "scores.txt"
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    output_file.write("Overall: %f\n" % final_mean)
    output_file.write("J-Mean: %f\n" % (np.mean(J_seen["M"])+np.mean(J_unseen["M"]))/2)
    output_file.write("J-seen-Mean: %f\n" % np.mean(J_seen["M"]))
    output_file.write("J-unseen-Mean: %f\n" % np.mean(J_unseen["M"]))
    output_file.write("F-Mean: %f\n" % (np.mean(F_seen["M"])+np.mean(F_unseen["M"]))/2)
    output_file.write("F-seen-Mean: %f\n" % np.mean(F_seen["M"]))
    output_file.write("F-unseen-Mean: %f\n" % np.mean(F_unseen["M"]))    
    output_file.write("V-Mean: %f\n" % (np.mean(V_seen["M"])+np.mean(V_unseen["M"]))/2)
    output_file.write("V-seen-Mean: %f\n" % np.mean(V_seen["M"]))
    output_file.write("V-unseen-Mean: %f\n" % np.mean(V_unseen["M"]))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
