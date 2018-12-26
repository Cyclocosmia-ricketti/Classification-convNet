from .model import cnn_AlexNet
from .data_load import *

# Define dir and hyperparam
batch_size = 50
epoches = 40
steps = 200
data_dir = '../dataset'
saving_dir = '../model'

# Load and Process dataset
print('load and process dataset...')
images, y_labels = load_TinyImagenet(data_dir)
images_batch,y_labels_batch = preprocess(images,y_labels,batch_size, steps)

# Setup model
print('setup model...')
alexnet = cnn_AlexNet(batch_size)

# Training
print('start training...')
for epoch in range(epoches):
    print('epoch: %d' % epoch)
    for step in range(steps):
        loss = alexnet.iteration(images_batch[step], y_labels_batch[step])
        print('epoch: %d , ' % epoch + 'step: %d , ' % step + 'Loss: %f' % loss)
print('finish training.')
alexnet.saving(saving_dir)
print('model saved.')



