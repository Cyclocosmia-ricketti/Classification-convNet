import numpy as np
from .model import cnn_AlexNet
from scipy.misc import imread

#test one image
def test(model, x):
    # Problem: unknown predicition
    # scores = model.prediction(x[None, :, :, :])
    scores = {}
    top_three = np.argsort(scores)[:, -3:]
    top_three = top_three.reshape(3)
    for i in range(3):
        print('ID:%d Confidence:%f'%(top_three[-1 - i], scores[top_three[-1 - i]]))

if __name__ == '__main__':
    model_dir = '../model/params.txt'
    model = cnn_AlexNet()
    model.restore(model_dir)
    image = imread('../test_img.jpg')
    test(model, image)