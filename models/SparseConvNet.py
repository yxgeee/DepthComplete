from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from models.SparseModule import SparseConvBlock


class SparseConvNet(nn.Module):

	def __init__(self, kernels=[11,7,5,3,3], mid_channel=16):
		super(SparseConvNet, self).__init__()
		channel = 1
		convs = []
		for i in range(len(kernels)):
			assert (kernels[i]%2==1)
			convs += [SparseConvBlock(channel, mid_channel, kernels[i], padding=(kernels[i]-1)//2)]
			channel = mid_channel
		self.sparse_convs = nn.Sequential(*convs)
		self.mask_conv = nn.Conv2d(mid_channel+1, 1, 1)

	def forward(self, x):
		m = (x>0).detach().float()
		x, m = self.sparse_convs((x,m))
		x = torch.cat((x,m), dim=1)
		x = self.mask_conv(x)
		# x = F.relu(x, inplace=True)
		return x
