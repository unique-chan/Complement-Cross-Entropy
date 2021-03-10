import time
import datetime
from torch import manual_seed, cuda, backends
import numpy as np
import random
import os
import csv


def write_log(my_args, my_trainer):
    # current time
    cur_time = datetime.datetime.now()
    cur_time = cur_time.strftime('%y-%m-%d-%H-%M-%S')
    # parser-arguments (namespace) -> dictionary
    new_my_args = parsed_arguments_dict(my_args)
    # make directory (my_logs)
    dir_name = 'my_logs'
    os.makedirs(dir_name, exist_ok=True)
    # csv log_file ->
    log_file = os.path.join(dir_name, 'logs.csv')
    fields_name = ['datetime', 'dataset_dir', 'network_name', 'loss_func',
                   'test_acc_best', 'valid_acc_best', 'train_acc_best',
                   'epochs', 'lr', 'lr_step', 'lr_step_gamma', 'lr_warmup_epochs',
                   'mean_std', 'height', 'width', 'batch_size', 'clip',
                   'train_acc_list', 'valid_acc_list', 'train_loss_list', 'valid_loss_list',
                   'train_acc_top5_list', 'valid_acc_top5_list', 'center_crop_size',
                   'minus_1_to_plus_1_rescale']
    mode = 'a' if os.path.exists(log_file) else 'w'
    # csv log_file write/update ->
    with open(log_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_name)
        if mode == 'w':
            writer.writeheader()
        del_candidate_keys = list(set(new_my_args.keys()) - set(fields_name))
        for key in del_candidate_keys:
            new_my_args.pop(key)
        # result_dic
        result_dic = new_my_args
        for key in result_dic.keys():
            if type(result_dic[key]) == list:
                result_dic[key] = str(result_dic[key])
        result_dic.update({'datetime': cur_time,
                           'train_acc_best': max(my_trainer.train_top1_acc_list),
                           'valid_acc_best': max(my_trainer.valid_top1_acc_list),
                           'test_acc_best': my_trainer.test_top1_acc,
                           'train_acc_list': str(my_trainer.train_top1_acc_list),
                           'valid_acc_list': str(my_trainer.valid_top1_acc_list),
                           'train_loss_list': str(my_trainer.train_loss_list),
                           'valid_loss_list': str(my_trainer.valid_loss_list),
                           'train_acc_top5_list': str(my_trainer.train_top5_acc_list),
                           'valid_acc_top5_list': str(my_trainer.valid_top5_acc_list)})
        writer.writerow(result_dic)


def parsed_arguments_dict(my_args):
    # my_args = Parser.parse_args()
    keys = [key for key in dir(my_args) if key[0] != '_']
    dict = {}
    for key in keys:
        dict[key] = eval('my_args.' + str(key))
    return dict


def fix_random_seed(seed=1234):
    # Ref.: https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True


def topk_acc(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # topk_acc_list = [correct[:k].view(-1).float().sum(0, keepdim=True) for k in topk]
    topk_acc_list = [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]
    return topk_acc_list


class ProgressBar:
    last_time = time.time()
    begin_time = last_time
    TOTAL_BAR_LENGTH = 10.

    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def progress_bar(front_msg, epoch_num, current_batch_idx, batch_size, msg=None):
        if current_batch_idx == 0:
            ProgressBar.last_time = time.time()
            ProgressBar.begin_time = ProgressBar.last_time
        cur_len = int(ProgressBar.TOTAL_BAR_LENGTH * current_batch_idx / batch_size)
        rest_len = int(ProgressBar.TOTAL_BAR_LENGTH - cur_len) - 1
        print('%s | Epoch: %5d [' % (front_msg, epoch_num), end='')
        for i in range(cur_len):
            print('■', end='')
        print('▶', end='')
        for i in range(rest_len):
            print(' ', end='')
        print(']', end='')
        cur_time = time.time()
        step_time = cur_time - ProgressBar.last_time
        ProgressBar.last_time = cur_time
        tot_time = cur_time - ProgressBar.begin_time
        msg_list = list()
        msg_list.append('  Step: %s' % ProgressBar.format_time(step_time))
        msg_list.append(' | Time: %s' % ProgressBar.format_time(tot_time))
        if msg:
            msg_list.append(' | ' + msg)
        msg = ''.join(msg_list)
        print(msg, end='')
        print(' | Batch: %d/%d \t\t' % (current_batch_idx + 1, batch_size), end='')
        if current_batch_idx < batch_size - 1:
            print('', end='\r')
        else:
            print('', end='\n')
