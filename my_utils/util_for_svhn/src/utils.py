# Copyright (C) 2018 Project AGI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging
import binascii
from PIL import Image

NUM_RANDOM = 6
NUM_LABELS = 10
DISPLAY_STEP = 1000

def logger_level(level):
    """
    Map the specified level to the numerical value level for the logger

    :param level: Logging level from command argument
    """
    try:
        level = level.lower()
    except AttributeError:
        level = ""

    return {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }.get(level, logging.WARNING)

def init_label_count():
    """
    Initialises the label count dictionary with 0 values
    """
    label_count = {}

    for i in range(NUM_LABELS):
        label_count[i] = 0

    return label_count

def generate_filename(dataset, label, label_count):
    """
    Generates a randomised filename for an image

    :param dataset:
    :param label: The groundtruth label of the image
    :param label_count: The number of times label has been seen already
    """
    random = binascii.hexlify(os.urandom(NUM_RANDOM // 2)).decode()
    filename = '%s_%s_%i_%i.png' % (dataset, random, label, label_count)
    return filename

def preprocess(dataset, features, labels, target_path, grayscale=False):
    """
    Map the specified level to the numerical value level for the logger

    :param dataset: The dataset type which could be train or test
    :param features: The features (X) of the dataset
    :param labels: The labels (y) of the dataset
    ;param target_path: The path to saving the file
    """
    label_count = init_label_count()

    # Get size of the data
    size = features.shape[3]

    for i in range(size):
        label = labels[i][0]

        count = 0
        if label in label_count:
            count = label_count[label]
            count += 1
        label_count[label] = count

        filename = generate_filename(dataset, label, label_count[label])
        filepath = os.path.join(target_path, filename)
        image = Image.fromarray(features[:, :, :, i])

        if grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image.save(filepath)

        if i % DISPLAY_STEP == 0 or i == 1:
            logging.info('Step #%i: saved %s', i, filename)
