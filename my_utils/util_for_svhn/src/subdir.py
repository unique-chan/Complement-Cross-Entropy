import argparse
import os
import shutil


def make_class_dir_and_move():
    pngs = os.listdir()

    for png in pngs:
        png_split = png.split('_')
        category = png_split[2]
        os.makedirs(category, exist_ok=True)
        shutil.move(png, os.path.join(category, png))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser = parser.parse_args()

    os.chdir(parser.input_path)
    print('current directory has been changed into:', os.getcwd())
    make_class_dir_and_move()
    print('done.')
