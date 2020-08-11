import os
import torch
from torch import optim
import argparse
import multiprocessing
import numpy as np
import itertools
import matplotlib.pyplot as plt
from torch.backends import cudnn

import get_model
import utils
import loader
from sklearn.metrics import confusion_matrix

# HYPER PARAMETERS
# if you run this .py file without initial augments, please set parameters manually!!!

# I. parsing arguments from .sh
PARSER = argparse.ArgumentParser(description='PyTorch Image Classification (github.com/unique-chan)')
utils.add_arguments_for_test(PARSER)
ARGS = PARSER.parse_args()

# II. dataset
DATASET_FOLDER_NAME = ARGS.dataset
DATASET_DIR = './a_datasets' + '/' + DATASET_FOLDER_NAME
TRAIN_DIR = DATASET_DIR + '/' + 'train'
VALID_DIR = DATASET_DIR + '/' + 'valid'
TEST_DIR = DATASET_DIR + '/' + 'test'
DATETIME = ARGS.datetime

# III. image resolution
IMAGE_HEIGHT = ARGS.height
IMAGE_WIDTH = ARGS.width

# IV. epochs, batch_size, num_classes
BATCH_SIZE = ARGS.batchsize
CLASSES = os.listdir(TRAIN_DIR)
NUM_CLASSES = len(CLASSES)

# V. training objective {"ERM", "COT", "CCE", "FL"}
OBJ_FUNC = ARGS.objective

# VI. pytorch-based CNN model
MODEL = get_model.get_model(ARGS.model_index)
MODEL_CHECKPOINT_DIR = './logs' + '/' + MODEL.__name__ + '/' + DATASET_FOLDER_NAME + '/' + OBJ_FUNC + '/' + DATETIME

# VII. num_workers for pytorch
# if you have a trouble with num_workers, set this variable to 0.
NUM_WORKERS = int(multiprocessing.cpu_count() / 4)


def get_max_pth_file():
    files = os.listdir(MODEL_CHECKPOINT_DIR)
    pths = []
    for file in files:
        dot_split = file.split('.')
        if len(dot_split) > 2 and dot_split[-1] == 'pth':
            pths.append(float(file[:-4]))
    pths.sort()
    return str(pths[-1]) + '.pth'


@torch.no_grad()
def get_all_y_hats_and_ys(model, test_loader, device):
    all_y_hats = torch.tensor([])
    all_ys = torch.tensor([], dtype=torch.long)
    if device == 'cuda':
        all_y_hats = all_y_hats.to(device)
        all_ys = all_ys.to(device)
    for batch in test_loader:
        x, ys = batch
        if device == 'cuda':
            x, ys = x.cuda(), ys.cuda()
        y_hat = model(x)
        all_y_hats = torch.cat((all_y_hats, y_hat), dim=0)
        all_ys = torch.cat((all_ys, ys), dim=0)
    if device == 'cuda':
        all_y_hats, all_ys = all_y_hats.to('cpu'), all_ys.to('cpu')
    return all_y_hats, all_ys


def modify_checkpoint_keys(chkpoint):
    keys = list(chkpoint['model_state_dict'].keys())
    for key in keys:
        new_key = '.'.join(key.split('.')[1:])
        chkpoint['model_state_dict'][new_key] = chkpoint['model_state_dict'].pop(key)
    return chkpoint


def plot_confusion_matrix(conf_matrix, classes, title=None, normalize=True,
                          cmap=plt.cm.Blues, file_format='svg'):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix:')
    else:
        print('confusion matrix:')
    print(conf_matrix)
    if title:
        plt.title(title)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    if len(classes) > 30:
        print('[WARNING] # of classes > 30, each class label will be not printed out.')
    else:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        threshold = conf_matrix.max() / 2.
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, format(conf_matrix[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment='center', color='white' if conf_matrix[i, j] > threshold else 'black')
    plt.tight_layout()
    plt.ylabel('Ground Truth Class' + ('' if len(classes) <= 30 else ' Index'))
    plt.xlabel('Predicted Class' + ('' if len(classes) <= 30 else ' Index'))
    conf_matrix_path = MODEL_CHECKPOINT_DIR + '/' + 'conf_matrix.%s' % file_format
    plt.savefig(conf_matrix_path, dpi=600)
    print('[RESULT] fig. of confusion matrix has been stored in (%s)' % conf_matrix_path)
    plt.show()


if __name__ == '__main__':
    # 1. Find test or valid directory.
    # if test directory doesn't exist, valid directory will used as the test directory.
    if not os.path.isdir(TEST_DIR):
        print('[WARNING] TEST_DIR (%s) does not exist. instead, VALID_DIR (%s) will be used for evaluation.'
              % (TEST_DIR, VALID_DIR))
        TEST_DIR = VALID_DIR
        if not os.path.isdir(TEST_DIR):
            print('[ERROR] neither TEST_DIR nor VALID_DIR exists. please check it.')
            exit(1)

    # 2. Create a test (valid) loader.
    _, test_loader = \
        loader.Loader.get_normalized_train_and_valid_loader(train_dir=None, valid_test_dir=TEST_DIR,
                                                            image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
                                                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # 3. Instantiate a CNN model.
    model = MODEL(num_classes=NUM_CLASSES)

    # 4. Initialize the CNN model with given checkpoint.
    checkpoint = torch.load(MODEL_CHECKPOINT_DIR + '/' + get_max_pth_file())
    checkpoint = modify_checkpoint_keys(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 5. Load the model to CPU or GPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 6. (If GPU,) Turn on 'Data Parallel' mode.
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # 7. Turn on the test (evaluation) mode.
    model.eval()

    # 8. Test (evaluate)
    y_hats, ys = get_all_y_hats_and_ys(model, test_loader, device)

    # 9. Result (1) (acc)
    top_acc_list = utils.topk_acc(y_hats, ys, topk=(1, 5))
    total_sample_sizes = utils.get_total_sample_sizes(TEST_DIR)
    print('[RESULT] Acc: [top-1]', 100 * top_acc_list[0].item() / total_sample_sizes, end=' ')
    print('[top-5]', 100 * top_acc_list[1].item() / total_sample_sizes)

    # 10. Result (2) (confusion matrix)
    conf_matrix = confusion_matrix(ys, y_hats.argmax(dim=1))
    plot_confusion_matrix(conf_matrix, CLASSES, file_format='svg')
