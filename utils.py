import sys
import time
import os
import csv

import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import _LRScheduler


# variables for progress_bar()
TOTAL_BAR_LENGTH = 25.
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


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def init_params(net):
    '''Init layer parameters.'''
    print('initialize model parameters.')
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def topk_acc(output, target, topk=(1,), sum_mode=True):
    '''
    topk = (1, )
    topk = (1, 5)
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        if sum_mode:
            res.append(correct_k)
        else: # 'avg_mode'
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_acc(preds, labels):
    '''get_num_correct'''
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_total_sample_sizes(dataset_dir):
    class_wise_sample_sizes = [len(os.listdir(dataset_dir + '/' + sub_dir)) for sub_dir in os.listdir(dataset_dir)]
    return sum(class_wise_sample_sizes)


def mkdir_for_logs(model, dataset_name, obj_func_name):
    resume_model_dir = './logs/' + model
    resume_dataset_dir = resume_model_dir + '/' + dataset_name
    resume_obj_dir = resume_dataset_dir + '/' + obj_func_name
    resume_time_dir = resume_obj_dir + '/' + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))

    resume_dirs = [resume_model_dir, resume_dataset_dir, resume_obj_dir, resume_time_dir]
    for resume_dir in resume_dirs:
        if not os.path.isdir(resume_dir):
            os.mkdir(resume_dir)

    resume_log_dir = resume_dirs[-1]
    return resume_log_dir


def csv_train_valid_writer_for_logs(log_dir):
    columns = ['epoch', 'loss', 'acc', 'top5acc', 'lr']

    train_file = open(log_dir + '/train.csv', 'w')
    valid_file = open(log_dir + '/valid.csv', 'w')

    train_writer = csv.DictWriter(train_file, fieldnames=columns)
    train_writer.writeheader()

    valid_writer = csv.DictWriter(valid_file, fieldnames=columns)
    valid_writer.writeheader()

    return train_file, train_writer, valid_file, valid_writer


def add_arguments_for_train(PARSER):
    PARSER.add_argument('--model_index', type=int,
                        help='network (refer to readme.md)')
    PARSER.add_argument('--dataset', type=str,
                        help='dataset folder name in "a_datasets"')
    PARSER.add_argument('--objective', default='ERM', type=str,
                        help='training objective function '
                             '(refer to readme.md) (default: ERM)')
    PARSER.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate (default: 0.1)')
    PARSER.add_argument('--epochs', default=200, type=int,
                        help='epoch (default: 200)')
    PARSER.add_argument('--batchsize', default=128, type=int,
                        help='batch size (default: 128)')
    PARSER.add_argument('--height', default=32, type=int,
                        help='height (default: 32)')
    PARSER.add_argument('--width', default=32, type=int,
                        help='width (default: 32)')
    PARSER.add_argument('--lrstep', default=[60, 60, 40, 40], type=list,
                        help='learning rate step decay milestones (default: [60, 60, 40, 40])')
    PARSER.add_argument('--lrstepgamma', default=0.2, type=float,
                        help='learning rate step decay gamma (default: 0.2)')
    PARSER.add_argument('--warmup', action='store_true',
                        help='initial learning rate warm up for first 5 epochs')
    PARSER.add_argument('--ccegamma', default=5, type=int,
                        help='gamma for CCE')
    PARSER.add_argument('--delay', default=0, type=int,
                        help='postpone training for given seconds (default: 0)')
    # PARSER.add_argument('--resume', '--r', action='store_true', help='resume the model on best acc')


def add_arguments_for_test(PARSER):
    PARSER.add_argument('--model_index', type=int,
                        help='network (refer to readme.md)')
    PARSER.add_argument('--dataset', type=str,
                        help='dataset folder name in "a_datasets"')
    PARSER.add_argument('--objective', type=str,
                        help='training objective function '
                             '(refer to readme.md)')
    PARSER.add_argument('--datetime', type=str,
                        help='datetime when you started to store your model parameters. '
                             'e.g. 2020-08-01-05-10-35')
    PARSER.add_argument('--batchsize', default=128, type=int,
                        help='batch size (default: 128)')
    PARSER.add_argument('--height', default=32, type=int,
                        help='height (default: 32)')
    PARSER.add_argument('--width', default=32, type=int,
                        help='width (default: 32)')

