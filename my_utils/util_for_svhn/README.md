Ref.: https://github.com/Cerenaut/Preprocess-SVHN

# Preprocess-SVHN
A tool for converting Google Street View House Number (SVHN) dataset into PNG images with additional preprocessing options such as grayscaling.

## Introduction
The tools provided are compatible with **Format 2** of the [SVHN](http://ufldl.stanford.edu/housenumbers/) which contains 32x32 cropped digits from the original images. There are 10 classes for this dataset (0-9), one for each digit.

Benchmarks for the SVHN dataset, and others can be found [here](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e).

### Preprocessing
The training and test datasets are provided in the `mat` format by the authors.
They are then loaded into Numpy arrays, and the features are separated from the labels.

The data is then converted to images into `training` and `testing` directories. The format
for the filename is as follows: `TYPE_RANDOM_LABEL_LABELCOUNT.png`

- `TYPE`: Indicates dataset type, could be either `train` or `test`
- `RANDOM`: Short randomly generated UUID-style characters e.g. `7daa28`
- `LABEL`: The groundtruth label for the image (between 0-9)
- `LABELCOUNT`: The count for how many times a label was seen to easily check for class balancing

This format is useful for quickly extracting information about the dataset and target labels from the filename, while ensuring that each image's filename is unique.

## Getting Started

### Requirements
- Python 2.7+

### Installation

Install the Python dependencies using pip: `pip install -r REQUIREMENTS.txt`

### Usage

Before starting, ensure that you have the `train_32x32.mat`, `test_32x32.mat` and `extra_32x32.mat` provided [here](http://ufldl.stanford.edu/housenumbers/). The script assumes the output directory exists so ensure that you have a designated output directory for the preprocessed images as it will not be created automatically.

To preprocess the training set, use the following:

`python src/svhn.py --dataset train --input_mat /path/to/train_32x32.mat --output_path /path/to/output/train`

Optionally, preprocess the additional training examples:

`python src/svhn.py --dataset train --input_mat /path/to/extra_32x32.mat --output_path /path/to/output/extra`

To preprocess the test set, use the following:

`python src/svhn.py --dataset test --input_mat /path/to/test_32x32.mat --output_path /path/to/output/test`

### png -> store into classwise subdirectories (Yechan added)

`python src/subdir.py --input_path /path/to/output/train`

`python src/subdir.py --input_path /path/to/output/extra`

`python src/subdir.py --input_path /path/to/output/test`

#### Grayscale

The original images are coloured, you may optionally pass the `--grayscale` parameter to convert the images to grayscale.

#### Logging

You may optionally pass the `--logging info` parameter to display the progress of the script, which looks like this:

```
...
[utils.py:80 - preprocess() - INFO] Step #6000: saved test_3d5a50_6_471.png
[utils.py:80 - preprocess() - INFO] Step #7000: saved test_4f1261_4_703.png
[utils.py:80 - preprocess() - INFO] Step #8000: saved test_a28782_10_543.png
[utils.py:80 - preprocess() - INFO] Step #9000: saved test_cb28b7_1_1752.png
[utils.py:80 - preprocess() - INFO] Step #10000: saved test_c7297a_9_621.png
[utils.py:80 - preprocess() - INFO] Step #11000: saved test_53fdfc_8_687.png
[utils.py:80 - preprocess() - INFO] Step #12000: saved test_0fb756_2_1932.png
[utils.py:80 - preprocess() - INFO] Step #13000: saved test_cca311_6_999.png
...
```
