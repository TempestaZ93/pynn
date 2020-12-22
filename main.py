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

    num_train = 5

    print("Reading labels")
    labels = read_labels('res/training/train-labels-idx1-ubyte', num_train)
    print("Reading images")
    images = read_images('res/training/train-images-idx3-ubyte', num_train)
    
    flat_images = _flat_images(images)
    print(np.max(flat_images))
    nn = Model([
    (784, 128, 'tanh'),
    (128, 10, 'softmax')],
    'log',
    0.05)

    nn.train(flat_images, labels, 1, 1)

    print(np.max(nn.layers[0].W))
    print(np.max(nn.layers[0].b))
    print(np.max(nn.layers[1].W))
    print(np.max(nn.layers[1].b))
    A = nn.feedforward(flat_images[0])
    print(labels[0])
    print(A)
    A = nn.feedforward(flat_images[1])
    print(labels[1])
    print(A)
    A = nn.feedforward(flat_images[2])
    print(labels[2])
    print(A)
    A = nn.feedforward(flat_images[3])
    print(labels[3])
    print(A)


if __name__ == "__main__":
    main(sys.argv[1:])