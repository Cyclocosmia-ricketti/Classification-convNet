import numpy as np
from model import cnn_AlexNet
from scipy.misc import imread

#test one image
def test(model, x):
    scores, _ = model.forward(x[None, :, :, :])
    top_three = np.argsort(scores)[:, -3:]
    top_three = top_three.reshape(3)
    for i in range(3):
        print('ID:%d Confidence:%f'%(top_three[-1 - i] + 1, scores[0, top_three[-1 - i]]))

if __name__ == '__main__':
    model_dir = '../model/params.txt'
    model = cnn_AlexNet()
    model.restore(model_dir)
    pre_dir = '../test/'
    while 1:
        print('please input image name:')
        name = input()
        if name == 'exit':
            break
        image = imread(pre_dir + name)
        image = image.transpose(2, 0, 1)
        test(model, image)
