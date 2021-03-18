import os
from keras.datasets import fashion_mnist
from PIL import Image, ImageOps


def save_image(file_name, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(file_name)


# Load Fashion-MNIST Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Make Directory
DIR_NAME = "Fashion_MNIST"
SUB_DIR_LIST = ['train', 'test']
os.makedirs(DIR_NAME, exist_ok=True)
for i in range(0, 10):
    for string in SUB_DIR_LIST:
        os.makedirs('{0}/{1}/{2}'.format(DIR_NAME, string, i), exist_ok=True)

# Save Images
GT_set = [y_train, y_test]
for i, img_set in enumerate([x_train, x_test]):
    for idx, img in enumerate(img_set):
        filename = "{0}/{1}/{2}/{3:05d}.jpg".format(DIR_NAME, SUB_DIR_LIST[i], GT_set[i][idx], idx)
        print(filename)
        save_image(filename, img)

print('done :)')