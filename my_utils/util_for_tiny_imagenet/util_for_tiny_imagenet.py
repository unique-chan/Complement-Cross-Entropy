import argparse
import os
import shutil
from glob import glob

'''
[How to use?]
e.g.
python3 util_for_tiny_imagenet.py --dir=/home/chan/PycharmProjects/tiny-imagenet-200
'''


'''
[Quick Overview of this source code]
<tiny-imagenet-200> directory-tree
train                             ➜ train
 ↳ n01443537                        ↳ n01443537
   ↳ images                             ↳ *.JPEG
     ↳ *.JPEG                       ↳ n01629819
 ↳ n01629819                            ↳ *.JPEG
   ↳ images                         ...
     ↳ *.JPEG
 ...

val                               ➜ valid
 ↳ images                           ↳ n01443537
   ↳ *.JPEG                             ↳ *.JPEG
 ↳ val_annotations.txt              ↳ n01629819
                                        ↳ *.JPEG
                                    ...
'''


# VALID
def valid_process():
    print('start to process the val (valid) directory.')
    valid_dir = 'val'
    valid_img_dir = os.path.join(valid_dir, 'images')
    valid_annot_file = open(os.path.join(valid_dir, 'val_annotations.txt'))
    valid_annots = valid_annot_file.readlines()

    for annot in valid_annots:
        annot_line = annot.split('\t')
        jpg, class_name = annot_line[0], annot_line[1]
        jpg_path, class_name_path = os.path.join(valid_img_dir, jpg), os.path.join(valid_dir, class_name)
        # make class_dir
        os.makedirs(class_name_path, exist_ok=True)
        # move val\images\*.JPEG    ->    val\[class]\*.JPEG
        print('move:', jpg_path, '->', os.path.join(class_name_path, jpg))
        shutil.move(jpg_path, os.path.join(class_name_path, jpg))

    # remove images (dir)
    shutil.rmtree(valid_img_dir)
    print('remove directory:', valid_img_dir)
    # rename val -> valid (dir)
    os.rename(valid_dir, 'valid')
    print('rename', valid_dir, '->', 'valid')


# TRAIN
def train_process():
    print('start to process the train directory.')
    train_dir = 'train'
    class_dirs = os.listdir(train_dir)

    for img_class in class_dirs:
        img_class_path = os.path.join(train_dir, img_class)
        jpgs_path = os.path.join(img_class_path, 'images')
        jpgs = os.listdir(jpgs_path)
        for jpg in jpgs:
            print('move:', os.path.join(jpgs_path, jpg), '->', os.path.join(img_class_path, jpg))
            shutil.move(os.path.join(jpgs_path, jpg), os.path.join(img_class_path, jpg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiny Imagenet Util (For Classification)')
    parser.add_argument('--dir', type=str)
    parser = parser.parse_args()

    tiny_imagenet = parser.dir
    os.chdir(tiny_imagenet)
    print('current directory has been changed into:', os.getcwd())

    train_process()
    valid_process()
    print('done.')
