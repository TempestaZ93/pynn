#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import read
import sys
import numpy as np
from model.model import Model
from input.reader import *


def _flat_images(images):
    flatted = np.zeros((np.size(images, 0), np.size(images, 1) * np.size(images, 2), 1))
    for idx in range(np.size(images, 0)):
        for row in range(np.size(images, 1)):
            for col in range(np.size(images, 2)):
                flatted[idx][row * np.size(images, 1) + col][0] = images[idx][row][col]

    return flatted


def main(argv):

    num_train = 2000

    print("Reading labels")
    labels = read_labels('res/training/train-labels-idx1-ubyte', num_train)
    print("Reading images")
    images = read_images('res/training/train-images-idx3-ubyte', num_train)
    
    flat_images = _flat_images(images)
    nn = Model([
    (784, 128, 'relu'),
    (128, 10, 'softmax')],
    'cross_entropy',
    0.1, 0.0001, 0.00001)

    nn.train(flat_images, labels, 10, 2)

    A = nn.feedforward(flat_images[0])
    print("Estimation:", A.argmax(), "| Ground Truth:", labels[0])
    A = nn.feedforward(flat_images[1])
    print("Estimation:", A.argmax(), "| Ground Truth:", labels[1])
    A = nn.feedforward(flat_images[2])
    print("Estimation:", A.argmax(), "| Ground Truth:", labels[2])
    A = nn.feedforward(flat_images[3])
    print("Estimation:", A.argmax(), "| Ground Truth:", labels[3])


if __name__ == "__main__":
    main(sys.argv[1:])