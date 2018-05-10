from __future__ import print_function, absolute_import
import os,sys
import os.path as osp

from utils.util import write_json, read_json

class Kitti(object):
	def __init__(self, root='./data/kitti', **kwargs):
		super(Kitti, self).__init__()
		self.root = root
		self.raw_split = 'velodyne_raw'
        self.gt_split = 'groundtruth'
		self.train_split = 'train'
		self.val_split = 'val'
		self.test_dir = 'depth_selection/test_depth_completion_anonymous/velodyne_raw'

		self.splits = osp.join(self.root, 'splits.json')
		if not self.splits.exists():
			_split_dataset()
		imgset = read_json(self.splits)
		self.trainset = {'raw':imgset['train_raw'], 'gt':imgset['train_gt']}
		self.valset = {'raw':imgset['val_raw'], 'gt':imgset['val_gt']}
		self.testset = {'raw':imgset['test_raw']}
        
	def _generate_list(self, split):
		raw = []
		gt = []
		path = osp.join(self.root, split)
		for subpath in os.listdir(path):
			for camid in os.listdir(osp.join(path,subpath,'proj_depth',self.raw_split)):
				for imgs in os.listdir(osp.join(path,subpath,'proj_depth',self.raw_split,camid)):
					raw.append(osp.join(split,subpath,'proj_depth',self.raw_split,camid,imgs))
					assert (osp.join(split,subpath,'proj_depth',self.gt_split,camid,imgs).exists())
					gt.append(osp.join(split,subpath,'proj_depth',self.gt_split,camid,imgs))
		return raw, gt

	def _split_dataset(self):
		train_raw, train_gt = _generate_list(self.train_split)
		val_raw, val_gt = _generate_list(self.val_split)
		test_raw = []
		for imgs in os.listdir(osp.join(self.root,self.test_dir)):
			test_raw.append(osp.join(self.test_dir,imgs))
		splits = {'train_raw':train_raw, 'train_gt':train_gt,
					'val_raw':val_raw, 'val_gt':val_gt,
					'test_raw':test_raw}
		write_json(splits, self.splits)

