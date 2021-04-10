import matplotlib.pyplot as plt
import numpy as np
import torch
from my_utils import parser, loader, model, util
import itertools
from sklearn.metrics import confusion_matrix

# if you encounter 'ModuleNotFoundError: No module named 'tkinter' in Linux,
# sudo apt-get install python3-tk


# Try to run this code as follows:
# 1/ python3 confusion_matrix.py --model_path='my_logs/21-04-10-12-25-23.pth' --network_name='resnet34_for_tiny' --dataset_dir='../CIFAR-10-lt-100' --height=32 --width=32  --mean_std
# And attempt to change this code for your purpose.


# Parser
my_parser = parser.Parser(mode='t-SNE')
my_args = my_parser.parse_args()
print('*', util.parsed_arguments_dict(my_args))

# Loader
my_loader = loader.Loader(my_args.dataset_dir, my_args.height, my_args.width,
                          my_args.batch_size, mean_std=my_args.mean_std, center_crop_size=my_args.center_crop_size,
                          minus_1_to_plus_1_rescale=my_args.minus_1_to_plus_1_rescale)
my_test_loader = my_loader.get_test_loader()

# Model
my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)
my_device = 'cuda:%d' % my_args.gpu_index
checkpoint = torch.load(my_args.model_path)
# print(my_args.network_name)
# print('checkpoint', checkpoint.keys())
my_model.load_state_dict(checkpoint)

my_model.to(my_device)
my_model.eval()

# Inference
with torch.no_grad():
    top1_acc, top5_acc = 0, 0
    progress_bar = util.ProgressBar()
    for batch_idx, (batch_input, batch_target) in enumerate(my_test_loader):
        batch_input, batch_target = batch_input.to(my_device), batch_target.to(my_device)
        output = my_model(batch_input)
        features = output.cpu() if batch_idx == 0 else np.concatenate((features, output.cpu()))
        targets = batch_target.cpu() if batch_idx == 0 else np.concatenate((targets, batch_target.cpu()))

        batch_top1_acc, batch_top5_acc = util.topk_acc(output, batch_target)
        top1_acc += batch_top1_acc
        top5_acc += batch_top5_acc

        progress_bar.progress_bar('test', 1, batch_idx, len(my_test_loader),
                                  msg='Acc.: [top1] %.3f%%, [top5] %.3f%%'
                                       % (top1_acc/len(targets) * 100, top5_acc/len(targets) * 100))


@torch.no_grad()
def get_all_y_hats_and_ys(model, test_loader, device):
    all_y_hats = torch.tensor([])
    all_ys = torch.tensor([], dtype=torch.long)
    if 'cuda' in device:
        all_y_hats = all_y_hats.to(device)
        all_ys = all_ys.to(device)
    for batch in test_loader:
        x, ys = batch
        if 'cuda' in device:
            x, ys = x.cuda(), ys.cuda()
        y_hat = model(x)
        all_y_hats = torch.cat((all_y_hats, y_hat), dim=0)
        all_ys = torch.cat((all_ys, ys), dim=0)
    if 'cuda' in device:
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
    global my_args
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
    plt.ylabel('Actual Class' + ('' if len(classes) <= 30 else ' Index'))
    plt.xlabel('Predicted Class' + ('' if len(classes) <= 30 else ' Index'))
    conf_matrix_path = '%s.%s' % (my_args.model_path.replace('/', ''), file_format)
    plt.savefig(conf_matrix_path, dpi=600)
    print('[RESULT] fig. of confusion matrix has been stored in (%s)' % conf_matrix_path)
    plt.show()


y_hats, ys = get_all_y_hats_and_ys(my_model, my_test_loader, my_device)
conf_matrix = confusion_matrix(ys, y_hats.argmax(dim=1))
plot_confusion_matrix(conf_matrix, my_loader.classes, file_format='svg')