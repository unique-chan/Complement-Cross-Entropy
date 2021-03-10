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

from __future__ import print_function

import os
import logging
import scipy.io as sio

import utils

def setup_arg_parsing():
    """
    Parse the commandline arguments
    """
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--dataset', dest='dataset', required=False,
                        help='The type of dataset could be (train|test|valid)')

    parser.add_argument('--input_mat', dest='input_path', required=True,
                        help='Path to the MAT source file')

    parser.add_argument('--output_path', dest='output_path', required=True,
                        help='Path to folder for saving generated images')

    parser.add_argument('--grayscale', dest='grayscale', action='store_true', required=False,
                        help='Convert images to grayscale (default=%(default)s).')

    parser.add_argument('--logging', dest='logging', required=False,
                        help='Logging level (default=%(default)s). '
                             'Options: debug, info, warning, error, critical')

    parser.set_defaults(grayscale=False)
    parser.set_defaults(logging='warning')

    return parser.parse_args()


def check_args(args):
    """
    Validates the arguments

    :param args: The commandline arguments
    """
    if not os.path.exists(args.input_path):
        logging.error('The input path is not valid: ' + args.input_path)
        exit(1)

    if not os.path.isdir(args.output_path):
        logging.error('The output path is not valid: ' + args.output_path)
        exit(1)


def main():
    """
    The main scope of the preprocessor containing the high level code
    """

    args = setup_arg_parsing()

    # Setup logging
    log_format = "[%(filename)s:%(lineno)s - %(funcName)s() - %(levelname)s] %(message)s"
    logging.basicConfig(format=log_format, level=utils.logger_level(args.logging))

    # Validate args
    check_args(args)

    # Load MAT source file
    try:
        data = sio.loadmat(args.input_path)
    except Exception as ex:
        logging.error('Failed to load input MAT file: ' + args.input_path)
        logging.error('Exception: %s', ex)
        exit(1)

    # Split data into features and labels
    features = data['X']
    labels = data['y']

    # Start preprocessing images
    utils.preprocess(args.dataset, features, labels, args.output_path, args.grayscale)

if __name__ == '__main__':
    main()
