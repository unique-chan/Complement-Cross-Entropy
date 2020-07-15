import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import multiprocessing

import torchvision.transforms as transforms
import torchvision.models as torch_models
from torchvision import datasets

from I_configuration import BATCH_SIZE, LEARNING_RATE, MODEL_INDEX, EPOCHS, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, \
    SAVE_CHECKPOINT_DIR, SAVE_CHECKPOINT_N_ITER, TRAIN_DIR, VALID_DIR, OBJ_FUNC_INDEX, GAMMA
from utils import progress_bar, init_params, get_mean_and_std_for_3ch_only, ComplementEntropy

# if you want to use your own CNN classifier models based on Pytorch,
# save the .py file in the models directory, and register your model in model_dic in get_model().
# for example, suppose you store resnet.py file in the models directory
# import models
# def get_model(idx):
#   model_dic = {
#       ...
#       55 : models.Resnet.Resnet18,
#       ...
#   }

from torch.utils.tensorboard import SummaryWriter


def get_model(idx):
    model_dic = {
        0: torch_models.alexnet,
        1: torch_models.vgg11,
        2: torch_models.vgg11_bn,
        3: torch_models.vgg13,
        4: torch_models.vgg13_bn,
        5: torch_models.vgg16,
        6: torch_models.vgg16_bn,
        7: torch_models.vgg19,
        8: torch_models.vgg19_bn,
        9: torch_models.resnet18,
        10: torch_models.resnet34,
        11: torch_models.resnet50,
        12: torch_models.resnet101,
        13: torch_models.resnet152,
        14: torch_models.resnext50_32x4d,
        15: torch_models.resnext101_32x8d,
        16: torch_models.wide_resnet50_2,
        17: torch_models.wide_resnet101_2,
        18: torch_models.squeezenet1_0,
        19: torch_models.squeezenet1_1,
        20: torch_models.densenet121,
        21: torch_models.densenet161,
        22: torch_models.densenet169,
        23: torch_models.densenet201,
        24: torch_models.inception_v3,
        25: torch_models.googlenet,
        26: torch_models.shufflenet_v2_x0_5,
        27: torch_models.shufflenet_v2_x1_0,
        28: torch_models.shufflenet_v2_x1_5,
        29: torch_models.shufflenet_v2_x2_0,
        30: torch_models.mobilenet_v2,
        31: torch_models.mnasnet0_5,
        32: torch_models.mnasnet0_75,
        33: torch_models.mnasnet1_0,
        34: torch_models.mnasnet1_3,
    }
    return model_dic[idx]


def get_obj_func_name(idx):
    obj_func_dic = {
        0: 'CE',
        1: 'COT',
        2: 'SCOT-1',
        3: 'SCOT-2'
    }
    return obj_func_dic[idx]


def get_transform(data_augmentation=False, mean=None, std=None, for_normalize=False):
    # for train
    if data_augmentation:
        train_flag = True
        transform_list = [
            transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    # for test or dev or init_normalization
    else:
        train_flag = False
        transform_list = [
            transforms.ToTensor()
        ]

    if not for_normalize:
        if type(mean) == torch.Tensor and type(std) == torch.Tensor:
            print('normalizing {%s} dataset ...' % ('training' if train_flag else 'valid'))
            transform_list += [transforms.Normalize(mean=mean, std=std)]

    return transforms.Compose(transform_list)


def get_train_loader(transform, shuffle=True, num_workers=0, for_normalize=False):
    if for_normalize:
        train_set = datasets.ImageFolder(root=TRAIN_DIR,
                                         transform=transform)
        return torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

    processed_train_set = datasets.ImageFolder(root=TRAIN_DIR,
                                               transform=transform)
    return torch.utils.data.DataLoader(processed_train_set, batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=num_workers)


def get_valid_loader(transform, shuffle=False, num_workers=0):
    processed_valid_set = datasets.ImageFolder(root=VALID_DIR,
                                               transform=transform)
    return torch.utils.data.DataLoader(processed_valid_set, batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    # tensorboard --logdir=runs
    model = get_model(MODEL_INDEX)  # class
    model_name = '%s-%s-EPOCHS%s-%s' \
                 % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    model.__name__,
                    EPOCHS,
                    get_obj_func_name(OBJ_FUNC_INDEX)
                    )
    model = model()                 # instance
    # init_params(model)              # init_params

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    number_workers = int(multiprocessing.cpu_count() / 4)

    print('model_name:', model_name)
    print('device:', device)

    # for normalizing to match the distribution of train, valid, and test data set.
    train_mean, train_std = \
        get_mean_and_std_for_3ch_only(
            get_train_loader(transform=get_transform(data_augmentation=False,
                                                     for_normalize=True),
                             shuffle=False, num_workers=number_workers,
                             for_normalize=True))

    train_dataloader = get_train_loader(transform=get_transform(data_augmentation=True,
                                                                mean=train_mean,
                                                                std=train_std),
                                        num_workers=number_workers)
    valid_dataloader = get_valid_loader(transform=get_transform(data_augmentation=False,
                                                                mean=train_mean,
                                                                std=train_std),
                                        num_workers=number_workers)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    log_dir = 'runs/%s' % model_name
    writer = SummaryWriter(log_dir=log_dir)

    iter_columns = ['iter', 'loss', 'acc', 'lr']
    epoch_columns = ['epoch', 'loss', 'acc', 'lr']

    iter_train_file = open(log_dir + '/iter-train.csv', 'w')
    iter_train_writer = csv.DictWriter(iter_train_file, fieldnames=iter_columns)
    iter_valid_file = open(log_dir + '/iter-valid.csv', 'w')
    iter_valid_writer = csv.DictWriter(iter_valid_file, fieldnames=iter_columns)

    epoch_train_file = open(log_dir + '/epoch-train.csv', 'w')
    epoch_train_writer = csv.DictWriter(epoch_train_file, fieldnames=epoch_columns)
    epoch_valid_file = open(log_dir + '/epoch-valid.csv', 'w')
    epoch_valid_writer = csv.DictWriter(epoch_valid_file, fieldnames=epoch_columns)

    primary_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
    #                       momentum=0.9, weight_decay=5e-4)

    complement_criterion = ComplementEntropy()
    complement_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                     factor=0.5, verbose=True, patience=10)


    def SCOT_criterion(outputs, targets, gamma):
        cross_entropy_loss = primary_criterion(outputs, targets)
        complement_entropy = complement_criterion(outputs, targets)

        return cross_entropy_loss + gamma * complement_entropy


    def train(epoch, CE=False, COT=False, SCOT1=False, SCOT2=False):
        model.train()
        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            optimizer.zero_grad()

            # for softmax activation in training
            outputs = nn.Softmax(dim=1)(outputs)

            if CE or COT:
                loss = primary_criterion(outputs, targets)

            if SCOT1:
                loss = SCOT_criterion(outputs, targets, gamma=GAMMA)

            if SCOT2:
                loss = complement_criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size()[0]
            correct += predicted.eq(targets).sum().item()

            progress_bar('Train', epoch + 1, batch_idx, len(train_dataloader),
                         msg='Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100.0 * (correct/total),
                            correct, total))

            if COT:
                entropy = complement_criterion(outputs, targets)

                complement_optimizer.zero_grad()
                entropy.backward()
                complement_optimizer.step()

            # if batch_idx % SAVE_CHECKPOINT_N_ITER == SAVE_CHECKPOINT_N_ITER - 1:
            #     writer.add_scalar('training loss', running_loss / SAVE_CHECKPOINT_N_ITER,
            #                       epoch * len(train_dataloader) + batch_idx)
            #     iter_train_writer.writerow({'iter': epoch * len(train_dataloader) + batch_idx,
            #                                 'loss': running_loss / SAVE_CHECKPOINT_N_ITER,
            #                                 'acc': 100.0 * (correct / total),
            #                                 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            #     running_loss = 0

            writer.add_scalar('training loss', running_loss,
                              epoch * len(train_dataloader) + batch_idx)
            iter_train_writer.writerow({'iter': epoch * len(train_dataloader) + batch_idx,
                                        'loss': running_loss,
                                        'acc': 100.0 * (correct / total),
                                        'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            running_loss = 0

        epoch_train_writer.writerow({'epoch': epoch,
                                     'loss': train_loss / len(train_dataloader),
                                     'acc': 100.0 * (correct / total),
                                     'lr': optimizer.state_dict()['param_groups'][0]['lr']})


    def valid(epoch):
        global best_acc
        model.eval()
        valid_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = primary_criterion(outputs, targets)

                valid_loss += loss.item()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size()[0]
                correct += predicted.eq(targets).sum().item()

                progress_bar('Valid', epoch + 1, batch_idx, len(valid_dataloader),
                             msg='Loss: %.3f | Acc.: %.3f%% (%d/%d)'
                             % (valid_loss / (batch_idx + 1), 100.0 * (correct/total),
                                correct, total))

                # if batch_idx % SAVE_CHECKPOINT_N_ITER == SAVE_CHECKPOINT_N_ITER - 1:
                #     writer.add_scalar('valid loss', running_loss / SAVE_CHECKPOINT_N_ITER,
                #                       epoch * len(valid_dataloader) + batch_idx)
                #     iter_valid_writer.writerow({'iter': epoch * len(valid_dataloader) + batch_idx,
                #                                 'loss': running_loss / SAVE_CHECKPOINT_N_ITER,
                #                                 'acc': 100.0 * (correct/total),
                #                                 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                #     running_loss = 0

                writer.add_scalar('valid loss', running_loss,
                                  epoch * len(valid_dataloader) + batch_idx)
                iter_valid_writer.writerow({'iter': epoch * len(valid_dataloader) + batch_idx,
                                            'loss': running_loss,
                                            'acc': 100.0 * (correct / total),
                                            'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                running_loss = 0

        acc = 100.0 * (correct / total)
        if acc > best_acc:
            print('Saving current status of the trained model (checkpoint) ...')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'loss': valid_loss,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, SAVE_CHECKPOINT_DIR + model_name + '.pth')
            best_acc = acc

        epoch_valid_writer.writerow({'epoch': epoch,
                                     'loss': valid_loss / len(valid_dataloader),
                                     'acc': acc,
                                     'lr': optimizer.state_dict()['param_groups'][0]['lr']})
        # learning rate scheduling
        scheduler.step(valid_loss)


    def cross_entropy_learning(epoch):
        train(epoch, CE=True)
        valid(epoch)


    def cot_learning(epoch):
        train(epoch, COT=True)
        valid(epoch)


    def scot_1_learning(epoch):
        train(epoch, SCOT1=True)
        valid(epoch)


    def scot_2_learning(epoch):
        train(epoch, SCOT2=True)
        valid(epoch)


    training_and_valid = cross_entropy_learning
    if OBJ_FUNC_INDEX == 0:
        training_and_valid = cross_entropy_learning
    elif OBJ_FUNC_INDEX == 1:
        training_and_valid = cot_learning
    elif OBJ_FUNC_INDEX == 2:
        training_and_valid = scot_1_learning
    elif OBJ_FUNC_INDEX == 3:
        training_and_valid = scot_2_learning
    else:
        print('Wrong OBJ_FUNC_INDEX! Please check it in I_configuration.py.')

    for epoch in range(0, EPOCHS):
        training_and_valid(epoch)

    iter_train_file.close()
    iter_valid_file.close()
    epoch_train_file.close()
    epoch_valid_file.close()
