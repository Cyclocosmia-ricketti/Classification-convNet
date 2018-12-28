from data_load import load_TinyImagenet, preprocess
from model import cnn_AlexNet
from update import adam
import numpy as np


def check_accuracy(model, x_batch, y_batch, batch_size=100):
    accuary = 0
    correct_num = 0
    N = x_batch.shape[0] 
    for i in range(N):
        scores, _ = model.forward(x_batch[i])
        top_three = np.argsort(scores)[:, -3:]
        correct_num += np.sum((y_batch - 1)[i] == top_three)

    accuary = correct_num / (N * batch_size)
    return accuary


# Define dir and hyperparam
batch_size = 50
epochs = 40
steps = 200
data_dir = '../dataset'
saving_dir = '../model'
adam_params = {}
loss_history = []
train_acc_history = []
lr_decay = 0.95

# Load and Process dataset
print('load and process dataset...')
images, y_labels = load_TinyImagenet(data_dir)
images_batch, y_labels_batch = preprocess(images, y_labels, batch_size, steps)

images_batch = images_batch.transpose(0, 1, 4, 2, 3)

# Setup model
print('setup model...')
alexnet = cnn_AlexNet(batch_size=batch_size)
for p in alexnet.params:
    adam_params[p] = None

# Training
print('start training...')
for epoch in range(epochs):
    for step in range(steps):
        loss, grads = alexnet.iteration(images_batch[step], y_labels_batch[step])
        loss_history.append(loss)
        print('epoch: %d , ' % (epoch+1) + 'step: %d , ' % (step+1) + 'Loss: %f' % loss)

        # Update parameters
        for p, w in alexnet.params.items():
            dw = grads[p]
            alexnet.params[p], adam_params[p] = adam(w, dw, adam_params[p])

    # decay the learning rate
    for p in adam_params:
        adam_params[p]['learning_rate'] *= lr_decay

    # check accuary
    train_acc = check_accuracy(alexnet, images_batch, y_labels_batch, batch_size=batch_size)
    train_acc_history.append(train_acc)
    print('Epoch %d/%d, train_acc: %f' % (
        epoch + 1, epochs, train_acc))

    # save parameters
    alexnet.saving(saving_dir)

print('finish training.')
alexnet.saving(saving_dir)
print('model saved.')