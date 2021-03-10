import os
import torch
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
import pickle


class Loader:
    mean_pkl, std_pkl = 'mean.pkl', 'std.pkl'

    def __init__(self, dataset_path, image_height, image_width, batch_size=128, num_workers=0,
                 mean_std=True, center_crop_size=0, minus_1_to_plus_1_rescale=False):
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(dataset_path, 'train')
        self.valid_dir = os.path.join(dataset_path, 'valid')
        self.test_dir = os.path.join(dataset_path, 'test')
        self.num_workers = num_workers
        self.num_classes = Loader.__get_num_classes(self.train_dir)
        self.image_height, self.image_width = image_height, image_width
        self.batch_size = batch_size
        self.mean_std = mean_std
        self.minus_1_to_plus_1_rescale = minus_1_to_plus_1_rescale
        self.center_crop_size = center_crop_size
        if mean_std:
            self.train_mean, self.train_std = self.get_train_mean_std()

    def get_train_mean_std(self):
        Loader.mean_pkl, Loader.std_pkl = os.path.join(self.dataset_path, Loader.mean_pkl), \
                                          os.path.join(self.dataset_path, Loader.std_pkl),
        if os.path.exists(Loader.mean_pkl) and os.path.exists(Loader.std_pkl):
            with open(Loader.mean_pkl, 'rb') as pkl:
                mean = pickle.load(pkl)
            with open(Loader.std_pkl, 'rb') as pkl:
                std = pickle.load(pkl)
        else:
            print('* Hold a sec for calculating mean/std of training examples.')
            loader = self.get_train_loader_for_mean_std()
            mean, std = torch.zeros(3), torch.zeros(3)
            for inputs, targets in loader:
                for i in range(3):
                    mean[i] += inputs[:, i, :, :].mean()
                    std[i] += inputs[:, i, :, :].std()
            mean /= len(loader)
            std /= len(loader)
            # memo for future use
            with open(Loader.mean_pkl, 'wb') as pkl:
                pickle.dump(mean, pkl)
            with open(Loader.std_pkl, 'wb') as pkl:
                pickle.dump(std, pkl)
        return mean, std

    @staticmethod
    def __get_num_classes(root_dir):
        return len([dir_ for dir_ in os.listdir(root_dir) if not os.path.isfile(dir_)])

    def get_train_transform(self):
        transforms_list = [
            transforms.RandomCrop((self.image_height, self.image_width), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()   # [HxWxC] [0, 255] -> [CxHxW] [0., 1.]
        ]
        if self.minus_1_to_plus_1_rescale:
            # [0., 1.] -> [-1., 1.]
            transforms_list.extend([transforms.Lambda(lambda x: x * 2. - 1.)])

        if self.mean_std:
            transforms_list.extend([transforms.Normalize(self.train_mean, self.train_std)])
        return transforms.Compose(transforms_list)

    def get_eval_transform(self):
        transforms_list = [
            transforms.ToTensor()   # [HxWxC] [0, 255] -> [CxHxW] [0., 1.]
        ]
        if self.center_crop_size > 0:
            transforms_list.insert(0, transforms.CenterCrop(self.center_crop_size))
        if self.mean_std:
            transforms_list.extend([transforms.Normalize(self.train_mean, self.train_std)])
        return transforms.Compose(transforms_list)

    def get_train_loader_for_mean_std(self):
        train_set = datasets.ImageFolder(root=self.train_dir, transform=transforms.ToTensor())
        return data.DataLoader(train_set, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self, shuffle=True):
        composed_transforms_list = self.get_train_transform()
        train_set = datasets.ImageFolder(root=self.train_dir, transform=composed_transforms_list)
        return data.DataLoader(train_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)

    def get_valid_loader(self, shuffle=False):
        valid_set = datasets.ImageFolder(root=self.valid_dir,
                                         transform=self.get_eval_transform())
        return data.DataLoader(valid_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)

    def get_test_loader(self, shuffle=False):
        test_set = datasets.ImageFolder(root=self.test_dir,
                                        transform=self.get_eval_transform())
        return data.DataLoader(test_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)
