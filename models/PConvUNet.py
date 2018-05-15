from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class PConv(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
		super(PConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)

		nn.init.normal_(self.conv.weight, 0.0, 0.02)

	def forward(self, input):
		x, m = input
		assert (x.size()==m.size())
		x = x * m
		x = self.conv(x)

		weights = torch.ones_like(self.conv.weight)
		m = F.conv2d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		m = torch.clamp(m, min=1e-5)

		x = x * (1. / m)
		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = (m>1e-5).float()
		return x, m

class PConvEncoder(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride=2, padding=1, dilation=1, bias=True, bn=True):
		super(PConvEncoder, self).__init__()
		self.sparse_conv = PConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.ifbn = bn
		if self.ifbn:
			self.bn = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input):
		x, m = input
		# m = m.expand_as(x)
		x, m = self.sparse_conv((x, m))
		if self.ifbn:
			x = self.bn(x)
		x = self.relu(x)
		return x, m

class PConvDecoder(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True, bn=True, nonlinear=True):
		super(PConvDecoder, self).__init__()
		self.up = nn.Upsample(scale_factor=2)
		self.sparse_conv = PConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.ifbn = bn
		if self.ifbn:
			self.bn = nn.BatchNorm2d(out_channel)
		self.if_nonlinear = nonlinear
		if self.if_nonlinear:
			self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, input, pre_input):
		x1, m1 = input
		x2, m2 = pre_input

		x1, m1 = self.up(x1), self.up(m1)
		m1 = (m1>0).float()
		x = torch.cat((x1, x2), dim=1)
		m = torch.cat((m1, m2), dim=1)

		x, m = self.sparse_conv((x, m))
		if self.ifbn:
			x = self.bn(x)
		if self.if_nonlinear:
			x = self.relu(x)
		return x, m

class UNet(nn.Module):

	def __init__(self, in_channel=1, out_channel=1):
		super(UNet, self).__init__()
		self.down1 = PConvEncoder(in_channel, 64, 11, stride=2, padding=5, bn=False)
		self.down2 = PConvEncoder(64, 128, 7, stride=2, padding=3, bn=True)
		self.down3 = PConvEncoder(128, 256, 5, stride=2, padding=2, bn=True)
		self.down4 = PConvEncoder(256, 512, 3, stride=2, padding=1, bn=True)
		self.down5 = PConvEncoder(512, 512, 3, stride=2, padding=1, bn=True)
		
		self.up1 = PConvDecoder(512+512, 512, 3, stride=1, padding=1, bn=True)
		self.up2 = PConvDecoder(512+256, 256, 3, stride=1, padding=1, bn=True)
		self.up3 = PConvDecoder(256+128, 128, 3, stride=1, padding=1, bn=True)
		self.up4 = PConvDecoder(128+64, 64, 3, stride=1, padding=1, bn=True)
		self.up5 = PConvDecoder(64+in_channel, out_channel, 3, stride=1, padding=1, bn=False, nonlinear=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		m = (x>0).detach().float()
		x1, m1 = self.down1((x, m))
		x2, m2 = self.down2((x1, m1))
		x3, m3 = self.down3((x2, m2))
		x4, m4 = self.down4((x3, m3))
		x5, m5 = self.down5((x4, m4))

		x9, m9 = self.up1((x5, m5), (x4, m4))
		x10, m10 = self.up2((x9, m9), (x3, m3))
		x11, m11 = self.up3((x10, m10), (x2, m2))
		x12, m12 = self.up4((x11, m11), (x1, m1))
		x13, _ = self.up5((x12, m12), (x, m))

		return x13


		
		
