import errno
import math
import os
import warnings

import numpy as np
from PIL import Image


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])



    return M, O, D

def db_statistics_var(per_frame_values_j,per_frame_values_f):
    JF=(per_frame_values_j+per_frame_values_f)/2
    JFM=np.nanmean(JF)
    value_len=JF.shape[1]
    var=(JFM-JF)
    V=(np.nansum((var**2))/value_len)
    V= round (V,4)
    V=math.sqrt(V)
    
    return V


