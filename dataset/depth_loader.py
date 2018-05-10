from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.

    return depth

class DepthDataset(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform
        # TODO transform: flip, scale, crop, eraser
        
    def __len__(self):
        return len(self.dataset['raw'])

    def __getitem__(self, index):
        raw_path = osp.join(self.root,self.dataset['raw'][index])
        gt_path = osp.join(self.root,self.dataset['gt'][index])
        raw = depth_read(raw_path)
        assert (raw.shape[0]==1216)
        assert (raw.shape[1]==352)
        gt = depth_read(gt_path)
        assert ((gt<0).sum()==0)
        
        if self.transform is not None:
            raw = self.transform(raw)
            gt = self.transform(gt)
        
        return raw, gt