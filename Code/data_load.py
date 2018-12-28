import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import random


def load_TinyImagenet(dataset_dir):
    """
    load dataset: Tiny ImageNet
    dim: 10000 * 64 * 64 * 3

    """
    X = []
    Y = []
    for i in range(1, 21):
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


def preprocess(images, y_labels, batch_size=50, steps = 200):
    # Minus average
    rgb_aver = np.mean(images, axis=0)
    rgb_aver = np.mean(rgb_aver, axis=0)
    rgb_aver = np.mean(rgb_aver, axis=0)
    rgb_aver = rgb_aver.repeat(10000 * 64 * 64).reshape([10000, 64, 64, 3])
    images = images - rgb_aver

    # Disorder order
    index = [i for i in range(len(y_labels))]
    random.shuffle(index)
    images = images[index]
    y_labels = y_labels[index]

    # cut into batch_size 200*50*64*64*3
    images_batch = images.reshape([-1, batch_size, 64, 64, 3])
    y_labels_batch = y_labels.reshape([-1, batch_size, 1])
    return images_batch, y_labels_batch
