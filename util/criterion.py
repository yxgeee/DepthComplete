import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedMAELoss(nn.Module):
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class MaskedLogMSELoss(nn.Module):
    def __init__(self):
        super(MaskedLogMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = torch.log(target+1e-6) - torch.log(pred+1e-6)
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedLogMAELoss(nn.Module):
    def __init__(self):
        super(MaskedLogMAELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = torch.log(target+1e-6) - torch.log(pred+1e-6)
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask].abs()

        delta = self.threshold * diff.max().item()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        self.loss = (part1 + part2).sum()
        return self.loss

__factory = {
    'masked_mseloss': MaskedMSELoss,
    'masked_maeloss': MaskedMAELoss,
    'masked_log_mseloss': MaskedLogMSELoss,
    'masked_log_maeloss': MaskedLogMAELoss,
    'berhuloss': BerHuLoss,
}

def get_criterions():
    return __factory.keys()

def init_criterion(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown criterion: {}".format(name))
    return __factory[name](**kwargs)
