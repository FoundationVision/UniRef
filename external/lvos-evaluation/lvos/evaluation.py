import sys
import warnings

from tqdm import tqdm
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)


import numpy as np
from lvos.lvos_seperate import LVOS
from lvos.metrics import db_eval_boundary, db_eval_iou
from lvos import utils
from lvos.results import Results
from scipy.optimize import linear_sum_assignment


class LVOSEvaluation(object):
    def __init__(self, lvos_root, task, gt_set, codalab=False):
        """
        Class to evaluate LVOS sequences from a certain set and for a certain task
        :param lvos_root: Path to the LVOS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.lvos_root = lvos_root
        self.task = task
        self.dataset = LVOS(root=lvos_root, task=task, subset=gt_set, codalab=codalab)

        sys.path.append(".")
        if codalab:
            self.unseen_videos=os.path.join(lvos_root,'unseen_videos.txt')
        else:
            self.unseen_videos='./unseen_videos.txt'

        self.unseen_videos=open(self.unseen_videos,mode='r').readlines()
        for vi in range(len(self.unseen_videos)):
            self.unseen_videos[vi]=self.unseen_videos[vi].strip()

    def _evaluate_semisupervised(self,seq,results, all_void_masks, metric):
        seq_name=list(seq.keys())[0]
        seq=seq[seq_name]

        objs=list(seq.keys())
        j_metrics_res=dict()
        f_metrics_res=dict()
        for oi in range(len(objs)):
            _obj=objs[oi]
            _frame_num=seq[_obj]['frame_range']['frame_nums']
            j_metrics_res[str(_obj)]=np.zeros((1,int(_frame_num)))
            f_metrics_res[str(_obj)]=np.zeros((1,int(_frame_num)))
        for oi in range(len(objs)):
            _obj=objs[oi]
            _frame_num=seq[_obj]['frame_range']['frame_nums']
            start_frame=seq[_obj]['frame_range']['start']
            end_frame=seq[_obj]['frame_range']['end']


            oidx=0
            for ii in range(int(start_frame),int(end_frame),5):
                gt_mask,_= self.dataset.get_mask(seq_name, "{0:08d}".format(ii),_obj)
                res_mask=results.read_mask(seq_name,"{0:08d}".format(ii),_obj)
                if 'J' in metric:
                    j_metrics_res[str(_obj)][0, oidx] = db_eval_iou(gt_mask, res_mask, all_void_masks)
                if 'F' in metric:
                    f_metrics_res[str(_obj)][0, oidx] = db_eval_boundary(gt_mask, res_mask, all_void_masks)
                oidx=oidx+1

        return j_metrics_res, f_metrics_res


    def evaluate(self, res_path, metric=('J', 'F', 'V'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')


        # Containers
        metrics_res = {}
        metrics_res_seen = {}
        metrics_res_unseen = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_seen['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_unseen['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_seen['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_unseen['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'V' in metric:
            metrics_res['V'] = {"M": [], "M_per_object": {}}
            metrics_res_seen['V'] = {"M": [], "M_per_object": {}}
            metrics_res_unseen['V'] = {"M": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            
            seq=self.dataset.get_sequence(seq)


            _seq_name=list(seq.keys())[0]
            objs=list(seq[_seq_name])
            is_unseen=False
            if _seq_name in self.unseen_videos:
                is_unseen=True
            if self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(seq,results, None, metric)
            for ii in range(len(objs)):
                _obj=objs[ii]
                seq_name = f'{_seq_name}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[str(_obj)])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                    if is_unseen:
                        metrics_res_unseen['J']["M"].append(JM)
                        metrics_res_unseen['J']["R"].append(JR)
                        metrics_res_unseen['J']["D"].append(JD)

                        metrics_res_unseen['J']["M_per_object"][seq_name] = JM

                    else:
                        metrics_res_seen['J']["M"].append(JM)
                        metrics_res_seen['J']["R"].append(JR)
                        metrics_res_seen['J']["D"].append(JD)

                        metrics_res_seen['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[str(_obj)])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM
                    if is_unseen:
                        metrics_res_unseen['F']["M"].append(FM)
                        metrics_res_unseen['F']["R"].append(FR)
                        metrics_res_unseen['F']["D"].append(FD)

                        metrics_res_unseen['F']["M_per_object"][seq_name] = FM

                    else:
                        metrics_res_seen['F']["M"].append(FM)
                        metrics_res_seen['F']["R"].append(FR)
                        metrics_res_seen['F']["D"].append(FD)

                        metrics_res_seen['F']["M_per_object"][seq_name] = FM

                if 'V' in metric and 'J' in metric and 'F' in metric: 
                    VM = utils.db_statistics_var(j_metrics_res[str(_obj)],f_metrics_res[str(_obj)])
                    metrics_res['V']['M']=VM
                    metrics_res['V']["M_per_object"][seq_name] = VM

                    if is_unseen:
                        metrics_res_unseen['V']["M"].append(VM)

                        metrics_res_unseen['V']["M_per_object"][seq_name] = VM
                    else:
                        metrics_res_seen['V']["M"].append(VM)
                        
                        metrics_res_seen['V']["M_per_object"][seq_name] = VM

            
        return metrics_res,metrics_res_seen,metrics_res_unseen
