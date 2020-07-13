import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import multiprocessing

import torchvision.transforms as transforms
from torchvision import datasets

from I_configuration import BATCH_SIZE, LEARNING_RATE, MODEL_INDEX, EPOCHS, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS
from utils import progress_bar

from models import resnet


def get_model(idx):
    model_dic = {
        29: resnet.ResNet18,
        30: resnet.ResNet34,
        31: resnet.ResNet50,
        32: resnet.ResNet101,
        33: resnet.ResNet152
    }
    return model_dic[idx]


def get_transform(data_augmentation):
    if data_augmentation:
        # for train
        return transforms.Compose([
            transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        # for test or dev
        return transforms.Compose([
            transforms.ToTensor()
        ])
    #         return transforms.ToTensor()


def get_train_loader(transform, shuffle=True, num_workers=0):
    # data_utils.Ten
    processed_train_set = datasets.ImageFolder(root='./B_dataset/train',
                                               transform=transform)
    return torch.utils.data.DataLoader(processed_train_set, batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=num_workers)


# def get_test_loader(transform, shuffle=False, num_workers=0):
#     processed_test_set = datasets.ImageFolder(root='./B_dataset/test',
#                                               transform=transform)
#     return torch.utils.data.DataLoader(processed_test_set, batch_size=BATCH_SIZE,
#                                        shuffle=shuffle, num_workers=num_workers)


def get_valid_loader(transform, shuffle=False, num_workers=0):
    processed_valid_set = datasets.ImageFolder(root='./B_dataset/valid',
                                               transform=transform)
    return torch.utils.data.DataLoader(processed_valid_set, batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    model = get_model(MODEL_INDEX)()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    number_workers = int(multiprocessing.cpu_count() / 4)

    train_dataloader = get_train_loader(transform=get_transform(data_augmentation=True),
                                        num_workers=number_workers)
    valid_dataloader = get_valid_loader(transform=get_transform(data_augmentation=False),
                                        num_workers=number_workers)
    # test_dataloader  = get_test_loader(transform=get_transform(data_augmentation=False),
    #                                    num_workers=number_workers)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # 모델 저장 관련
    # 구현해야 함.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=0.9, weight_decay=5e-4)

    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print('inputs.size()', inputs.size())
            # print('targets.size()', targets.size())
            # print('targets:', targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            # for softmax activation in training
            outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print('loss.item', loss.item())
            train_loss += loss.item()
            # print('train_loss', train_loss)
            _, predicted = outputs.max(1)

            # print('predicted', predicted)
            total += targets.size()[0]
            # print('targets', targets)
            correct += predicted.eq(targets).sum().item()

            progress_bar('Train', epoch + 1, batch_idx, len(train_dataloader),
                         msg='Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100.0 * (correct/total),
                            correct, total))

    def valid(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size()[0]
                correct += predicted.eq(targets).sum().item()

                progress_bar('Valid', epoch + 1, batch_idx, len(valid_dataloader),
                             msg='Loss: %.3f | Acc.: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100.0 * (correct/total),
                                correct, total))

        acc = 100.0 * (correct / total)
        if acc > best_acc:
            print('Saving current status of the trained model ...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    for epoch in range(0, EPOCHS):
        train(epoch)
        valid(epoch)
