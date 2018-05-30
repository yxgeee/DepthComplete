from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import math
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

class Evaluate(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.rmse, self.mae = 0, 0
        self.log_rmse, self.log_mae = 0, 0
        self.absrel, self.sqrel = 0, 0
        self.num = 0

    def pipline(self):
        return self.rmse

    # def update(self, result):
    #     self.irmse, self.imae = result.irmse, result.imae
    #     self.rmse, self.mae = result.rmse, result.mae
    #     self.log_rmse, self.log_mae = result.log_rmse, result.log_mae
    #     self.absrel, self.sqrel = result.absrel, result.sqrel

    def evaluate(self, output, target):
        valid_mask = (target>0).detach()
        self.num = valid_mask.sum().item()
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()
        abs_log_diff = (torch.log(output+1e-6) - torch.log(target+1e-6)).abs()

        self.rmse = math.sqrt((torch.pow(abs_diff, 2)).mean())
        self.mae = abs_diff.mean()
        self.log_rmse = math.sqrt((torch.pow(abs_log_diff, 2)).mean())
        self.log_mae = abs_log_diff.mean()
        self.absrel = (abs_diff / target).mean()
        self.sqrel = (torch.pow(abs_diff, 2) / (torch.pow(target, 2))).mean()

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = abs_inv_diff.mean()

cmap = plt.cm.viridis
def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def dense_to_sparse(depth, num_samples):
    mask_keep = depth > 0
    n_keep = np.count_nonzero(mask_keep)
    prob = float(self.num_samples) / n_keep
    return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)