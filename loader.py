import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms


class Loader:
    @staticmethod
    def __get_transform(image_height, image_width,
                        data_augmentation=False, mean=None, std=None, for_normalize=False):
        # for train
        if data_augmentation:
            train_flag = True
            transform_list = [
                transforms.RandomCrop((image_height, image_width), padding=4),
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
                print('normalize %s dataset.' % ('training' if train_flag else 'valid (or test)'))
                transform_list += [transforms.Normalize(mean=mean, std=std)]

        return transforms.Compose(transform_list)

    @staticmethod
    def __get_train_loader(transform, train_dir, batch_size, shuffle=True, num_workers=0, for_normalize=False):
        if for_normalize:
            train_set = datasets.ImageFolder(root=train_dir,
                                             transform=transform)
            return torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        else:
            processed_train_set = datasets.ImageFolder(root=train_dir,
                                                       transform=transform)
            return torch.utils.data.DataLoader(processed_train_set, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def __get_valid_loader(transform, valid_dir, batch_size, shuffle=False, num_workers=0):
        processed_valid_set = datasets.ImageFolder(root=valid_dir,
                                                   transform=transform)
        return torch.utils.data.DataLoader(processed_valid_set, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def __get_mean_and_std_for_3ch_only(train_loader, msg='train'):
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('compute mean and std:')
        for inputs, targets in train_loader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(train_loader))
        std.div_(len(train_loader))
        print('\t▶ %s_mean:' % msg, mean)
        print('\t▶ %s_std: %s.' % (msg, std))
        return mean, std

    @staticmethod
    def get_normalized_train_and_valid_loader(train_dir=None, valid_test_dir=None, image_height=32, image_width=32,
                                              batch_size=128, num_workers=0):
        if train_dir is None and valid_test_dir is None:
            print('[ERROR] both train_dir and valid_dir are None.')
            exit(1)
        test_mode = False
        if train_dir is None:
            print('test (evaluation) mode is turned on.')
            test_mode = True
        store_dir = '/'.join(valid_test_dir.split('/')[:-1]) + '/' + 'mean_std.txt'
        if os.path.isfile(store_dir):
            f = open(store_dir, 'r')
            train_mean = torch.Tensor([float(element) for element in f.readline().replace('\n', '').split(',')])
            train_std = torch.Tensor([float(element) for element in f.readline().replace('\n', '').split(',')])
            print('\t▶ stored_train_mean:', train_mean)
            print('\t▶ stored_train_std:', train_std)
        else:
            train_mean, train_std = Loader.__get_mean_and_std_for_3ch_only(
                Loader.__get_train_loader(transform=Loader.__get_transform(image_height=image_height,
                                                                           image_width=image_width,
                                                                           data_augmentation=False,
                                                                           for_normalize=True),
                                          shuffle=False, num_workers=num_workers, for_normalize=True,
                                          train_dir=train_dir, batch_size=batch_size))

            # memorize calculated train_mean and train_std in *.txt file for further trainings.
            f = open(store_dir, 'w')
            f.write(str(train_mean.tolist())[1:-1] + '\n')
            f.write(str(train_std.tolist())[1:-1] + '\n')
        f.close()
        train_loader = None
        if not test_mode:
            train_loader = Loader.__get_train_loader(transform=Loader.__get_transform(image_height=image_height,
                                                                                      image_width=image_width,
                                                                                      data_augmentation=True,
                                                                                      mean=train_mean, std=train_std),
                                                     num_workers=num_workers, train_dir=train_dir,
                                                     batch_size=batch_size)

        valid_loader = Loader.__get_valid_loader(transform=Loader.__get_transform(image_height=image_height,
                                                                                  image_width=image_width,
                                                                                  data_augmentation=False,
                                                                                  mean=train_mean, std=train_std),
                                                 num_workers=num_workers, valid_dir=valid_test_dir, batch_size=batch_size)
        return train_loader, valid_loader
