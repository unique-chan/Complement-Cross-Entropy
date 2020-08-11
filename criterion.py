import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ComplementEntropy(nn.Module):
    '''Compute the complement entropy of complement classes.'''
    def __init__(self, num_classes=100):
        super(ComplementEntropy, self).__init__()
        self.classes = num_classes
        self.batch_size = None

    def forward(self, y_hat, y):
        self.batch_size = len(y)
        y_hat = F.softmax(y_hat, dim=1)
        Yg = torch.gather(y_hat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7
        Px = y_hat / Yg_.view(len(y_hat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_\
            (1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        entropy = torch.sum(output)
        entropy /= float(self.batch_size)
        entropy /= float(self.classes)
        return entropy


class ComplementCrossEntropy(nn.Module):
    def __init__(self, num_classes=100, gamma=5):
        super(ComplementCrossEntropy, self).__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.complement_entropy = ComplementEntropy(num_classes)

    def forward(self, y_hat, y):
        l1 = self.cross_entropy(y_hat, y)
        l2 = self.complement_entropy(y_hat, y)
        return l1 + self.gamma * l2


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_classes:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_classes, alpha=[0.25, 0.75], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_classes
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in (0,1).'
            assert balance_index > -1
            alpha = torch.ones((self.num_classes,))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Invalid alpha type, expect [int|float|list|tuple|torch.Tensor].')

    def forward(self, y_hat, y):
        # y_hat would be a logit.
        if y_hat.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            y_hat = y_hat.view(y_hat.size(0), y_hat.size(1), -1)
            y_hat = y_hat.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            y_hat = y_hat.view(-1, y_hat.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        y = y.view(-1, 1) # [N,d1,d2,...] -> [N*d1*d2*...,1]

        pt = y_hat.gather(1, y).view(-1) + self.eps # avoid apply
        log_pt = pt.log()

        if self.alpha.device != log_pt.device:
            alpha = self.alpha.to(log_pt.device)
            alpha_class = alpha.gather(0, y.view(-1))
            log_pt = alpha_class * log_pt

        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * log_pt
        loss = loss.mean() if self.size_average else loss.sum()
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1-alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, y_hat, y):
#         if y_hat.dim() > 2:
#             y_hat = y_hat.view(y_hat.size(0), y_hat.size(1), -1)  # N,C,H,W => N,C,H*W
#             y_hat = y_hat.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             y_hat = y_hat.contiguous().view(-1, y_hat.size(2))   # N,H*W,C => N*H*W,C
#         y = y.view(-1, 1)
#
#         log_pt = F.log_softmax(y_hat)
#         log_pt = log_pt.gather(1, y)
#         log_pt = log_pt.view(-1)
#         pt = Variable(log_pt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=y_hat.data.type():
#                 self.alpha = self.alpha.type_as(y_hat.data)
#             at = self.alpha.gather(0, y.data.view(-1))
#             log_pt = log_pt * Variable(at)
#
#         loss = -1 * (1 - pt) ** self.gamma * log_pt
#         return loss.mean() if self.size_average else loss.sum()
