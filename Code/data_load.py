import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
import pandas as pd

def load_TinyImagenet(dataset_dir):
    """
    load dataset: Tiny ImageNet
    dim: 10000 * 64 * 64 * 3

    """
    X = []
    Y = []
    for i in range(1, 22):
        image_dir = dataset_dir + '/' + str(i) + '/' + 'images'
        images = os.listdir(image_dir)
        for each_image in images:
            image = imread(image_dir + '/' + each_image)
            if len(image.shape) == 2:
                image = [image, image, image]
                image = np.array(image).transpose(1, 2, 0)
            X.append(image)
            Y.append(i)
    X = np.array(X).astype("float") / 255.0
    Y = np.array(Y)
    return X, Y


def preprocess(images):
    # Minus average
    rgb_aver = np.mean(images, axis=0)
    rgb_aver = np.mean(rgb_aver, axis=0)
    rgb_aver = np.mean(rgb_aver, axis=0)
    print(rgb_aver)
    rgb_aver = np.tile(rgb_aver, 11000 * 64 * 64).reshape([11000, 64, 64, 3])
    images = images - rgb_aver

    # # Disorder order
    # index = [i for i in range(len(y_labels))]
    # random.shuffle(index)
    # images = images[index]
    # y_labels = y_labels[index]

    # # cut into batch_size 200*50*64*64*3
    # images_batch = images.reshape([-1, batch_size, 64, 64, 3])
    # y_labels_batch = y_labels.reshape([-1, batch_size, 1])
    return images


def validation_preprocess(images, y_labels, batch_size=50):
    # Get size
    N = np.shape(images)[0]
    # Minus average
    rgb_aver = np.array([0.49398646, 0.45837655, 0.3890597])
    rgb_aver = np.tile(rgb_aver, N * 64 * 64).reshape([N, 64, 64, 3])
    images = images - rgb_aver

    # Disorder order
    index = [i for i in range(len(y_labels))]
    random.shuffle(index)
    # images = images[index]
    # y_labels = y_labels[index]
    images_batch = images.reshape([-1, batch_size, 64, 64, 3])
    y_labels_batch = y_labels.reshape([-1, batch_size, 1])
    return images_batch, y_labels_batch

def load_test(test_dir):
    X = []
    image_dir = test_dir + '/' + 'images'
    label_dir = test_dir + '/' + 'label.txt'
    label = pd.read_table(label_dir, header=None, encoding='utf-8', delim_whitespace=True, index_col=0)
    Y = np.array(label[1])
    images = os.listdir(image_dir)
    #images.sort(key=lambda x: int(x[4]))
    for each_image in images:
        image = imread(image_dir + '/' + each_image)
        if len(image.shape) == 2:
            image = [image, image, image]
            image = np.array(image).transpose(1, 2, 0)
        X.append(image)
    X = np.array(X).astype("float") / 255.0
    return X, Y
