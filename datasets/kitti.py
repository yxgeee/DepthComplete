from __future__ import print_function, absolute_import
import os,sys
import os.path as osp

from util.utils import write_json, read_json

class Kitti(object):
	def __init__(self, root='./data/kitti'):
		super(Kitti, self).__init__()
		self.root = root
		self.raw_split = 'velodyne_raw'
		self.gt_split = 'groundtruth'
		self.train_split = 'train'
		self.val_split = 'val'
		self.val_selected_split = 'depth_selection/val_selection_cropped'
		self.test_dir = 'depth_selection/test_depth_completion_anonymous/velodyne_raw'

		self.splits = osp.join(self.root, 'splits.json')
		if not osp.isfile(self.splits):
			self._split_dataset()
		imgset = read_json(self.splits)
		self.trainset = {'raw':imgset['train_raw'], 'gt':imgset['train_gt']}
		self.valset = {'raw':imgset['val_raw'], 'gt':imgset['val_gt']}
		self.valset_select = {'raw':imgset['val_selected_raw'], 'gt':imgset['val_selected_gt']}
		self.testset = {'raw':imgset['test_raw']}

	def _generate_list(self, split):
		raw = []
		gt = []
		path = osp.join(self.root, split)
		for subpath in os.listdir(path):
			for camid in os.listdir(osp.join(path,subpath,'proj_depth',self.raw_split)):
				for imgs in os.listdir(osp.join(path,subpath,'proj_depth',self.raw_split,camid)):
					raw.append(osp.join(split,subpath,'proj_depth',self.raw_split,camid,imgs))
					assert (osp.isfile(osp.join(path,subpath,'proj_depth',self.gt_split,camid,imgs)))
					gt.append(osp.join(split,subpath,'proj_depth',self.gt_split,camid,imgs))
		return raw, gt

	def _generate_list_selected(self, split):
		raw = []
		for imgs in sorted(os.listdir(osp.join(self.root, split))):
			raw.append(osp.join(split,imgs))
		return raw

	def _split_dataset(self):
		train_raw, train_gt = self._generate_list(self.train_split)
		val_raw, val_gt = self._generate_list(self.val_split)
		val_selected_raw = self._generate_list_selected(osp.join(self.val_selected_split,'velodyne_raw'))
		val_selected_gt = self._generate_list_selected(osp.join(self.val_selected_split,'groundtruth_depth'))
		test_raw = self._generate_list_selected(self.test_dir)
		splits = {'train_raw':train_raw, 'train_gt':train_gt,
					'val_raw':val_raw, 'val_gt':val_gt,
					'val_selected_raw':val_selected_raw, 'val_selected_gt':val_selected_gt,
					'test_raw':test_raw}
		write_json(splits, self.splits)

