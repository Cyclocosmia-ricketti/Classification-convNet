from data_load import load_TinyImagenet, preprocess, validation_preprocess
from model import cnn_VGGLike
from update import adam
import numpy as np
import os
import random
import pickle
def check_accuracy(model, x_batch, y_batch, batch_size=50):
    accuary = 0
    correct_num = 0
    N = x_batch.shape[0] 
    for i in range(N):
        scores, _ = model.forward(x_batch[i],'test')
        top_three = np.argsort(scores)[:, -3:]
        correct_num += np.sum((y_batch - 1)[i] == top_three)

    accuary = correct_num / (N * batch_size)
    return accuary


# Define dir and hyperparam
batch_size = 50
epochs = 30
steps = 220
data_dir = '../dataset'
saving_dir = '../model'
adam_params = {}
loss_history = []
train_acc_history = []
val_acc_history = []
lr_decay = 0.95

# Load and Process dataset
print('load and process dataset...')
images, y_labels = load_TinyImagenet(data_dir)
images= preprocess(images)
#images_batch = images_batch.transpose(0, 1, 4, 2, 3)

#load and Process validation set
val_dir = '../validation'
val_images, val_y_labels = load_TinyImagenet(val_dir)
val_images_batch, val_y_labels_batch = validation_preprocess(val_images, val_y_labels)
val_images_batch = val_images_batch.transpose(0, 1, 4, 2, 3)
# Setup model
print('setup model...')
vggnet = cnn_VGGLike(batch_size=batch_size)
for p in vggnet.params:
    adam_params[p] = None

# Training
print('start training...')
if (os.path.exists(saving_dir + '/' + 'params.txt')):
    print('restoring model')
    vggnet.restore(saving_dir + '/' + 'params.txt')

for epoch in range(epochs):
    # Disorder order
    index = [i for i in range(len(y_labels))]
    random.shuffle(index)
    images = images[index]
    y_labels = y_labels[index]
    # cut into batch_size 200*50*64*64*3
    images_batch = images.reshape([-1, batch_size, 64, 64, 3])
    y_labels_batch = y_labels.reshape([-1, batch_size, 1])
    images_batch = images_batch.transpose(0, 1, 4, 2, 3)
    for step in range(steps):
        loss, grads = vggnet.iteration(images_batch[step], y_labels_batch[step])
        loss_history.append(loss)
        print('epoch: %d , ' % epoch + 'step: %d , ' % step + 'Loss: %f' % loss)
        fp = open('./loss.txt', 'a')
        print('epoch: %d , ' % epoch + 'step: %d , ' % step + 'Loss: %f' % loss,file=fp)
        fp.close()
        # Update parameters
        for p, w in vggnet.params.items():
            dw = grads[p]
            vggnet.params[p], adam_params[p] = adam(w, dw, adam_params[p])

    # decay the learning rate
    for p in adam_params:
        adam_params[p]['learning_rate'] *= lr_decay

    # check accuary
    train_acc = check_accuracy(vggnet, images_batch, y_labels_batch, batch_size=batch_size)
    train_acc_history.append(train_acc)
    val_acc = check_accuracy(vggnet, val_images_batch, val_y_labels_batch, batch_size=batch_size)
    val_acc_history.append(val_acc)
    print('Epoch %d/%d, train_acc: %f, val_acc: %f' % (
        epoch + 1, epochs, train_acc, val_acc))

    # save parameters
    vggnet.saving(saving_dir, option=epoch)

    # write log
    fp = open('./log.txt', 'a')
    print('Epoch %d/%d, train_acc: %f, val_acc: %f' % (epoch + 1, epochs, train_acc, val_acc), file=fp)
    fp.close()
print('finish training.')
vggnet.saving(saving_dir)
print('model saved.')
