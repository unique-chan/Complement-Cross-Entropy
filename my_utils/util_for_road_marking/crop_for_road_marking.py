import os
import glob
from PIL import Image
import shutil
import random


path = '../../../RoadmarkingDataset/RoadMarkingDataset'
print(os.getcwd())
annotation = glob.glob(os.path.join(path, 'dataset_annotations.txt'))[0]
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'valid')
test_dir = os.path.join(path, 'test')


f = open(annotation, mode='r')
class_stat = {}

i = 0
while True:
    line = f.readline()
    if not line: break

    line = line[:-1]
    line = line.split(',')

    x1, y1 = int(float(line[0])), int(float(line[1]))
    x2, y2 = int(float(line[2])), int(float(line[3]))
    x3, y3 = int(float(line[4])), int(float(line[5]))
    x4, y4 = int(float(line[6])), int(float(line[7]))
    class_name = line[8]
    file_name = line[-1].replace('.png', '.jpg')

    img = Image.open(os.path.join(path, file_name))

    min_x, min_y = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
    max_x, max_y = max(x1, x2, x3, x4), max(y1, y2, y3, y4)
    w, h = max_x - min_x, max_y - min_y
    area = (min_x, min_y, min_x + w, min_y + h)

    if class_stat.get(class_name) is None:
        class_stat[class_name] = 1
    else:
        class_stat[class_name] += 1

    cropped_img = img.crop(area)
    cropped_img = cropped_img.resize((64, 64))
    class_path = os.path.join(path, class_name)
    os.makedirs(class_path, exist_ok=True)
    cropped_img.save(os.path.join(class_path, '{}.jpg'.format(i)), format='JPEG', quality=100)

    i += 1
f.close()

del_classes = ['diamond', 'car', 'pool', 'lane', 'keep', 'clear', '25', 'X-crossing',
                      'forward&right', 'speed', 'hump', '30', 'slow', 'school', 'stripe', 'forward&left', 'yield']

for del_class in del_classes:
    class_path = os.path.join(path, del_class)
    shutil.rmtree(class_path) ; del class_stat[del_class]

train_ratio = 0.8 ; valid_ratio = 0.1 ; test_ratio = 0.1

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

idx = 1
for k, v in class_stat.items():
    class_dir = os.path.join(path, k)
    samples = os.listdir(class_dir)
    set_samples = set(samples)

    # train_set
    train_no = int(v * train_ratio)
    train_samples = random.sample(samples, train_no)
    set_samples = set(samples) - set(train_samples)
    samples = list(set_samples)

    # valid_set
    valid_no = int(v * valid_ratio)
    valid_samples = random.sample(samples, valid_no)
    set_samples = set(samples) - set(valid_samples)
    samples = list(set_samples)

    # test_set
    test_no = v - train_no - valid_no
    test_samples = samples

    print('{}. class:'.format(idx), k, '=> train : valid : test = {} : {} : {}'.format(train_no, valid_no, test_no))

    train_class_dir = os.path.join(train_dir, k)
    os.makedirs(train_class_dir, exist_ok=True)
    for sample in train_samples:
        shutil.copy(os.path.join(os.path.join(path, k), sample), os.path.join(train_class_dir, sample))

    valid_class_dir = os.path.join(valid_dir, k)
    os.makedirs(valid_class_dir, exist_ok=True)
    for sample in valid_samples:
        shutil.copy(os.path.join(os.path.join(path, k), sample), os.path.join(valid_class_dir, sample))

    test_class_dir = os.path.join(test_dir, k)
    os.makedirs(test_class_dir, exist_ok=True)
    for sample in test_samples:
        shutil.copy(os.path.join(os.path.join(path, k), sample), os.path.join(test_class_dir, sample))

    idx += 1
    # print(train_samples, valid_samples, test_samples)

print('done :)')