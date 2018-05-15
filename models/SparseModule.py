from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class SparseConv(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
		super(SparseConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		mc = m.expand_as(x)
		x = x * mc
		x = self.conv(x)

		weights = torch.ones_like(self.conv.weight)
		mc = F.conv2d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc
		x = x * mc
		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = self.pool(m)

		return x, m

class SparseConvBlock(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
		super(SparseConvBlock, self).__init__()
		self.sparse_conv = SparseConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input):
		x, m = input
		x, m = self.sparse_conv((x, m))
		assert (m.size(1)==1)
		x = self.relu(x)
		return x, m

class SparseAdd(nn.Module):
	# Add sparse data
	def __init__(self, weighted=True):
		super(SparseAdd, self).__init__()
		self.weighted = weighted

	def forward(self, x1, x2, m1, m2):
		assert (x1.size()==x2.size())
		if not self.weighted:
			return x1+x2

		x = (x1 * m1.expand_as(x1) + x2 * m2.expand_as(x2)) / (m1.expand_as(x1) + m2.expand_as(x2))
		m = m1 + m2
		m[m>0] = 1
		return x, m





