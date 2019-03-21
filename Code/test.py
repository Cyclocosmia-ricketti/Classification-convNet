import numpy as np
from model import cnn_VGGLike
from scipy.misc import imread
from data_load import load_TinyImagenet, validation_preprocess, preprocess, load_test

#test one image
def test(model, x):
    scores, _ = model.forward(x[None, :, :, :],'test')
    top_three = np.argsort(scores)[:, -3:]
    top_three = top_three.reshape(3)
    for i in range(3):
        print('ID:%d Confidence:%f'%(top_three[-1 - i] + 1, scores[0, top_three[-1 - i]]))

def check_accuracy(model, x_batch, y_batch, batch_size=50):
    accuary = 0
    correct_num = 0
    N = x_batch.shape[0] 
    for i in range(N):
        scores, _ = model.forward(x_batch[i],'test')
        top_three = np.argsort(scores)[:, -3:]
        correct_num += np.sum((y_batch - 1)[i] == top_three)
        #current_acc = np.sum((y_batch - 1)[i] == top_three) / batch_size
        #print("acc=%f"%current_acc)
    accuary = correct_num / (N * batch_size)
    return accuary

if __name__ == '__main__':
    model_dir = '../model/params_final.txt'
    model = cnn_VGGLike()
    model.restore(model_dir)
    print('please select mode')
    mode = input()
    if mode == 'test_all':
        data_dir = '../test'
        print('load and process dataset...')
        images, y_labels = load_test(data_dir)
        images_batch, y_labels_batch = validation_preprocess(images, y_labels, batch_size=50)
        images_batch = images_batch.transpose(0, 1, 4, 2, 3)
        print('checking...')
        accuary = check_accuracy(model, images_batch, y_labels_batch, batch_size=50)
        print('test accuracy: %f' % accuary)
    if mode == 'val' :
        data_dir = '../validation'
        print('load and process dataset...')
        images, y_labels = load_TinyImagenet(data_dir)
        images_batch, y_labels_batch = validation_preprocess(images, y_labels)
        images_batch = images_batch.transpose(0, 1, 4, 2, 3)
        print('checking...')
        accuary = check_accuracy(model, images_batch, y_labels_batch)
        print('validation accuracy: %f' % accuary)