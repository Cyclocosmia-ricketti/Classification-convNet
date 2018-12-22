import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    images, tag = load_TinyImagenet('../dataset')
    first_image = images[0]
    first_tag = tag[0]
    plt.imshow(first_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
