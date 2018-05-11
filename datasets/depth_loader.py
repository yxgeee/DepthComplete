from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def depth_transform(pil_img):
    depth_png = np.array(pil_img, dtype=int)[:,:,np.newaxis]
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

class DepthDataset(Dataset):
    def __init__(self, root, dataset, height, width, isVal=False):
        self.root = root
        self.dataset = dataset
        self.height = height
        self.width = width
        self.isVal = isVal
        self.totensor = T.ToTensor()
        # TODO transform: flip, scale/crop, eraser
        
    def __len__(self):
        return len(self.dataset['raw'])

    def transform(self, raw, gt):
        # Random crop
        if not self.isVal:
            angle = np.random.uniform(-5.0, 5.0)
            raw = TF.rotate(raw, angle, resample=Image.NEAREST)
            gt = TF.rotate(raw, angle, resample=Image.NEAREST)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            raw, output_size=(self.height, self.width))
        if self.isVal:
            i, j = 0, 0
        raw = TF.crop(raw, i, j, h, w)
        gt = TF.crop(gt, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5 and not self.isVal:
            raw = TF.hflip(raw)
            gt = TF.hflip(gt)
        return raw, gt

    def __getitem__(self, index):
        raw_path = osp.join(self.root,self.dataset['raw'][index])
        gt_path = osp.join(self.root,self.dataset['gt'][index])

        raw_pil = Image.open(raw_path)
        gt_pil = Image.open(gt_path)
        raw_pil, gt_pil = self.transform(raw_pil, gt_pil)

        raw = depth_transform(raw_pil)
        gt = depth_transform(gt_pil)

        # scale to [0,1]
        # scale = raw.max()
        scale = 1
        gt_s = gt / scale
        raw_s = raw / scale
        gt_s[gt<0] = -1
        raw_s[raw<0] = -1

        raw_s = self.totensor(raw_s).float()
        gt_s = self.totensor(gt_s).float()
        
        return raw_s, gt_s, scale*256.