import torch
from torch import nn
import torch.nn.functional as F

# Reference: https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss_test.py


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, yHat, y):
        if yHat.dim() > 2:
            yHat = yHat.view(yHat.size(0), yHat.size(1), -1)  # N, C, H, W  => N, C, H*W
            yHat = yHat.transpose(1, 2)                         # N, C, H*W   => N, H*W, C
            yHat = yHat.contiguous().view(-1, yHat.size(2))    # N, H*W, C   => N*H*W, C
        y = y.view(-1, 1)

        logpt = F.log_softmax(yHat, dim=1)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != yHat.data.type():
                self.alpha = self.alpha.type_as(yHat.data)
            at = self.alpha.gather(0, y.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

