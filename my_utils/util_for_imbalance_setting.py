import os
import numpy as np


def lt_distribution(max_class_examples, class_num, imbalance_ratio):
    mu = np.power(1 / imbalance_ratio, 1 / (class_num - 1))
    list_of_example_num_per_class = [int(max_class_examples * np.power(mu, i)) for i in range(class_num)]
    return list(list_of_example_num_per_class)


def step_distribution(max_class_examples, class_num, imbalance_ratio):
    class_num_tuple = int(np.round(class_num / 2)), class_num - int(np.round(class_num / 2))
    min_class_examples = int(max_class_examples / imbalance_ratio)
    list_of_example_num_per_class = [max_class_examples for _ in range(class_num_tuple[0])] + \
                                    [min_class_examples for _ in range(class_num_tuple[1])]
    return list(list_of_example_num_per_class)


def make_imbalanced_train(path, distribution):
    class_dir_list = sorted(os.listdir(path))
    assert len(class_dir_list) == len(distribution), "len(class_dir_list) should be equal to len(distribution)."
    for i, class_dir in enumerate(class_dir_list):
        class_path = os.path.join(path, class_dir)
        img_file_list = sorted([img for img in os.listdir(class_path)
                                if os.path.isfile(os.path.join(class_path, img)) and
                                img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
        for j in range(len(img_file_list) - distribution[i]):
            img = os.path.join(class_path, img_file_list[j])
            # print(img, 'has been removed.')
            os.remove(img)


# print('distribution', lt_distribution(5000, 10, 10))
# print('distribution', lt_distribution(5000, 10, 100))
# print('=' * 50)

print('distribution:', step_distribution(500, 200, 10))
make_imbalanced_train('/home/chan/PycharmProjects/tiny-imagenet-200-step-10/train', step_distribution(500, 200, 10))
