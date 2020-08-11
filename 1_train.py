import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import argparse
import multiprocessing
import warnings

import get_model
import utils
import criterion
import loader

# HYPER PARAMETERS
# if you run this .py file without initial augments, please set parameters manually!!!

# I. parsing arguments from .sh
PARSER = argparse.ArgumentParser(description='PyTorch Image Classification (github.com/unique-chan)')
utils.add_arguments_for_train(PARSER)
ARGS = PARSER.parse_args()

# II. dataset
DATASET_FOLDER_NAME = ARGS.dataset
DATASET_DIR = './a_datasets' + '/' + DATASET_FOLDER_NAME
TRAIN_DIR = DATASET_DIR + '/' + 'train'
VALID_DIR = DATASET_DIR + '/' + 'valid'

# III. image resolution
IMAGE_HEIGHT = ARGS.height
IMAGE_WIDTH = ARGS.width

# IV. epochs, batch_size, num_classes
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batchsize
NUM_CLASSES = len(os.listdir(VALID_DIR))

# V. for manual learning rate scheduling
LEARNING_RATE = ARGS.lr
LR_STEP_MILESTONES = ARGS.lrstep
LR_STEP_GAMMA = ARGS.lrstepgamma

# VI. for linear learning rate warm-up scheduling
WARM_UP_EPOCHS = 5 if ARGS.warmup else 0

# VII. training objective {"ERM", "COT", "CCE", "FL"}
OBJ_FUNC = ARGS.objective

# VIII. pytorch-based CNN model
MODEL = get_model.get_model(ARGS.model_index)

# IX. num_workers for pytorch
# if you have a trouble with num_workers, set this variable to 0.
NUM_WORKERS = int(multiprocessing.cpu_count() / 4)

# X. if you want to postpone training a model, set TRAINING_DELAY > 0.
# e.g. TRAINING_DELAY = 7200 <=> 7200 sec = 2 hours
TRAINING_DELAY = ARGS.delay


def train(epoch, warm_up=True):
    model.train()
    train_loss = 0
    total, top1_correct, top5_correct = 0, 0, 0
    top1_acc, top5_acc = 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # learning rate warm up
        if warm_up and epoch <= WARM_UP_EPOCHS:
            warm_up_scheduler.step()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if OBJ_FUNC == 'FL':
            outputs = F.softmax(outputs, dim=1)

        optimizer.zero_grad()
        # loss_param = {'y_hat': outputs, 'y': targets} ; loss = primary_criterion(**loss_param)
        loss = primary_criterion(outputs, targets)

        if torch.isinf(loss):
            print('[ERROR] nan loss (%s), stop training.' % loss)
            exit(1)

        loss.backward(retain_graph=True)
        if OBJ_FUNC == 'COT':
            entropy = complement_criterion(outputs, targets)
            entropy.backward()

        optimizer.step()

        train_loss += loss.item()

        # _, predicted = outputs.max(1)
        total += targets.size()[0]

        top_acc_list = utils.topk_acc(outputs, targets, topk=(1, 5))

        # top1_correct += predicted.eq(targets).sum().item()
        top1_correct += top_acc_list[0]
        top5_correct += top_acc_list[1]

        top1_acc = 100.0 * (top1_correct / total)
        top5_acc = 100.0 * (top5_correct / total)

        utils.progress_bar('Train', epoch + 1, batch_idx, len(train_loader),
                           msg='Loss: %.3f | Acc: [top-1] %.3f%% (%d/%d), [top-5] %.3f%% (%d/%d)'
                               % (train_loss / (batch_idx + 1), top1_acc, top1_correct, total,
                                  top5_acc, top5_correct, total))

    train_loss /= len(train_loader)

    tensor_board_writer.add_scalars('train/acc', {'acc': top1_acc}, epoch)
    tensor_board_writer.add_scalars('train/loss', {'loss': train_loss}, epoch)
    train_csv_writer.writerow({'epoch': epoch, 'loss': train_loss, 'acc': top1_acc.item(),
                               'top5acc': top5_acc.item(), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
    train_csv_file.flush()


def valid(epoch):
    global best_top1_valid_acc
    model.eval()

    valid_loss = 0
    total, top1_correct, top5_correct = 0, 0, 0
    top1_acc, top5_acc = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = cross_entropy(outputs, targets)
            valid_loss += loss.item()

            total += targets.size()[0]
            top_acc_list = utils.topk_acc(outputs, targets, topk=(1, 5))
            top1_correct += top_acc_list[0]
            top5_correct += top_acc_list[1]

            top1_acc = 100.0 * (top1_correct / total)
            top5_acc = 100.0 * (top5_correct / total)

            utils.progress_bar('Valid', epoch + 1, batch_idx, len(valid_loader),
                               msg='Loss: %.3f | Acc: [top-1] %.3f%% (%d/%d), [top-5] %.3f%% (%d/%d)'
                                   % (valid_loss / (batch_idx + 1), top1_acc, top1_correct, total,
                                      top5_acc, top5_correct, total))

    if top1_acc > best_top1_valid_acc:
        print('Saving current parameters of the model (checkpoint).')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': top1_acc,
            'top5acc': top5_acc,
            'loss': valid_loss,
        }

        # 감마 얘기 추가해야 .
        torch.save(state, log_dir + '/' + str(top1_acc.item()) + '.pth')
        best_top1_valid_acc = top1_acc

    valid_loss /= len(valid_loader)

    tensor_board_writer.add_scalars('valid/acc', {'acc': top1_acc}, epoch)
    tensor_board_writer.add_scalars('valid/loss', {'loss': valid_loss}, epoch)
    valid_csv_writer.writerow({'epoch': epoch, 'loss': valid_loss, 'acc': top1_acc.item(),
                               'top5acc': top5_acc.item(), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
    valid_csv_file.flush()


def get_primary_crieterion(obj_func='ERM'):
    if obj_func in ['ERM', 'COT']:
        return cross_entropy
    elif obj_func in ['CCE']:
        return criterion.ComplementCrossEntropy(num_classes=NUM_CLASSES, gamma=ARGS.ccegamma)
    elif obj_func in ['FL']:
        return criterion.FocalLoss(num_classes=NUM_CLASSES, gamma=2.0, alpha=0.25, balance_index=2)
    else:
        print('[ERROR] invalid OBJ_FUNC (%s).' % obj_func)
        exit(1)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 0. Postpone training for TRAINING_DELAY seconds.
    time.sleep(TRAINING_DELAY)

    # 1. Create a directory for resume (model) parameters, CSV logs, and tensor-board summary logs.
    log_dir = utils.mkdir_for_logs(MODEL.__name__, DATASET_FOLDER_NAME, OBJ_FUNC)

    # 2. Instantiate a tensorboard writer and CSV writers.
    tensor_board_writer = SummaryWriter(log_dir=log_dir)
    train_csv_file, train_csv_writer, valid_csv_file, valid_csv_writer = \
        utils.csv_train_valid_writer_for_logs(log_dir=log_dir)

    # 3. Instantiate and initialize a CNN model.
    model = MODEL(num_classes=NUM_CLASSES)
    utils.init_params(model)

    # 4. Load the model to CPU or GPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 5. (If GPU,) Turn on 'Data Parallel' mode.
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print('model_info:', MODEL.__name__, '/ device:', device, '/ obj.:', OBJ_FUNC, '/ dataset:', DATASET_FOLDER_NAME)

    # 6. Load the train and valid dataset.
    train_loader, valid_loader = \
        loader.Loader.get_normalized_train_and_valid_loader(train_dir=TRAIN_DIR, valid_test_dir=VALID_DIR,
                                                            image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
                                                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 7. Set model's optimizer.
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    # 8. Set model's criterion (loss function).
    # typically, primary_criterion will be same as cross-entropy for 'ERM' and 'CCE'.
    # complement_criterion exists for 'COT'.
    cross_entropy = nn.CrossEntropyLoss()
    primary_criterion = get_primary_crieterion(obj_func=OBJ_FUNC)
    complement_criterion = criterion.ComplementEntropy(num_classes=NUM_CLASSES)

    # 9. Set model's learning rate scheduler.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_STEP_GAMMA)
    warm_up_scheduler = utils.WarmUpLR(optimizer, len(train_loader) * WARM_UP_EPOCHS)

    # 10. Train and valid the model.
    best_top1_valid_acc = 0
    for current_epoch in range(0, EPOCHS):
        train(current_epoch, warm_up=ARGS.warmup)
        valid(current_epoch)

    print('[RESULT]', 'model_info:', MODEL.__name__, '/ device:', device, '/ obj.:', OBJ_FUNC,
          '* (valid) best_top1_acc:', best_top1_valid_acc.item(), '/ dataset:', DATASET_FOLDER_NAME)
