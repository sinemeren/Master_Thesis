import torch
from torch.nn.modules.loss import L1Loss


def predictionErrorMSE(output, target):

    return torch.nn.MSELoss()(output, target)


def predictionError(output, target):
    err = abs(output - target)/target
    torch.mean(torch.mean(err, 1))
    #print("err",     torch.mean(torch.mean(err, 1))*100)
    return torch.nn.MSELoss()(output, target)
