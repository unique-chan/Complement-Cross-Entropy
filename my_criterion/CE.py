import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplementEntropy(nn.Module):
    def __init__(self, device):
        # self.C = C                                      # C (number of classes)
        self.device = device                            # {'cpu', 'cuda:0', 'cuda:1', ...}
        super(ComplementEntropy, self).__init__()

    def forward(self, yHat, y):
        batch_size = len(y)
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7                           # numerical trick
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0)
        # y_zerohot = torch.ones(batch_size, self.C).scatter_(
            # 1, y.view(batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device=self.device)
        loss = torch.sum(output)
        loss /= float(batch_size)
        loss /= float(yHat.shape[1])
        # loss /= float(self.C)
        return loss