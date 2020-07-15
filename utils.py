'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import torch
import sys
import time

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from I_configuration import NUM_CLASSES


class ComplementEntropy(nn.Module):
    '''Compute the complement entropy of complement classes.'''
    def __init__(self):
        super(ComplementEntropy, self).__init__()
        self.classes = NUM_CLASSES
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


class SimpleComplementEntropy1(nn.Module):
    '''Compute the complement entropy of complement classes.'''
    def __init__(self):
        super(SimpleComplementEntropy1, self).__init__()
        self.classes = NUM_CLASSES
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


def get_mean_and_std_for_3ch_only(train_data_loader, msg='train'):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('computing mean and std...')
    for inputs, targets in train_data_loader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(train_data_loader))
    std.div_(len(train_data_loader))
    print('▶ %s_mean:' % msg, mean)
    print('▶ %s_std:' % msg, std)
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


TOTAL_BAR_LENGTH = 50.
last_time = time.time()
begin_time = last_time


def progress_bar(front_msg, epoch_num, current_batch_idx, batch_size, msg=None):
    global last_time, begin_time
    if current_batch_idx == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current_batch_idx / batch_size)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write('%s | Epoch: %5d [' % (front_msg, epoch_num))
    for i in range(cur_len):
        sys.stdout.write('■')
    sys.stdout.write('▶')
    for i in range(rest_len):
        sys.stdout.write(' ')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    msg_list = list()
    msg_list.append('  Step: %s' % format_time(step_time))
    msg_list.append(' | Total Time: %s' % format_time(tot_time))
    if msg:
        msg_list.append(' | ' + msg)

    msg = ''.join(msg_list)
    sys.stdout.write(msg)

    sys.stdout.write(' | Batch: %d/%d ' % (current_batch_idx + 1, batch_size))

    if current_batch_idx < batch_size - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms '
        i += 1
    if f == '':
        f = '0ms'
    return f
