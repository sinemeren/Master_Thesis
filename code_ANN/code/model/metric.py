import torch
from torch.nn.modules.loss import L1Loss


def MSE(output, target):

    return torch.nn.MSELoss()(output, target)


def MAE(output, target):
    return torch.nn.L1Loss()(output, target)
