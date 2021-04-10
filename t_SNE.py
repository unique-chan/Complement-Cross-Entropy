import matplotlib.pyplot as plt
import numpy as np
# import torch
# from matplotlib import cm
# from sklearn.decomposition import PCA
import torch
from sklearn.manifold import TSNE
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from my_utils import parser, loader, model, util
from my_utils import parser, loader, model, util

# from visualization import ANN

# if you encounter 'ModuleNotFoundError: No module named 'tkinter' in Linux,
# sudo apt-get install python3-tk

# Ref.1: https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
# Ref.2: https://learnopencv.com/t-sne-for-feature-visualization/?fbclid=IwAR1OXjfEGlX8dazwX2FE8k3yZfujuCqZIalwb9CW8v2SqOfKX19bFrEPCn8


# This code is only compatible with 'resnet34_for_tiny_and_t_sne' (network) & 'Fashion-MNIST' (dataset).
# Try to run this code as follows:
# 1/
# 2/
# 3/
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
features, targets = None, None

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


# Scaling
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


tsne = TSNE(n_components=2).fit_transform(features)
tx, ty = tsne[:, 0], tsne[:, 1]
# tx, ty = scale_to_01_range(tx), scale_to_01_range(ty)

fig = plt.figure()
ax = fig.add_subplot(111)


colors_per_class = {
    'T-shirt/top' : [254, 202, 87],
    'Trouser' : [255, 107, 107],
    'Pullover' : [10, 189, 227],
    'Dress' : [255, 159, 243],
    'Coat' : [16, 172, 132],
    'Sandal' : [128, 80, 128],
    'Shirt' : [87, 101, 116],
    'Sneaker' : [52, 31, 151],
    'Bag' : [0, 0, 0],
    'Ankle boot' : [100, 100, 255],
}

for label_idx, label in enumerate(colors_per_class):
    indices = [idx for idx, target in enumerate(targets) if target == label_idx]
    current_tx, current_ty = np.take(tx, indices), np.take(ty, indices)

    color = np.array(colors_per_class[label], dtype=np.float) / 255.
    ax.scatter(current_tx, current_ty, color=color, alpha=0.5, label=label)
ax.legend(loc='best')
plt.show()
