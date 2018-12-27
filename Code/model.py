# coding = utf-8

import numpy as np
from .layers import *
import pickle
import os


class cnn_AlexNet(object):
    def __init__(self, batch_size=50, input_dim=(3, 32, 32), num_filters=(32), filter_size=(7),
                 hidden_dim=100, num_classes=20, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
            
        self.batch_size = batch_size
        C, H, W = input_dim 
        F1, HH1, WW1 = num_filters[0], filter_size[0], filter_size[0]

        self.params = {}
        self.params['w_conv1'] = weight_scale * np.random.randn(F1, C, HH1, WW1)
        self.params['b_conv1'] = np.zeros(F1)
        self.params['w_fc1'] = weight_scale * np.random.randn(F1*(H//2)*(W//2), hidden_dim)
        self.params['b_fc1'] = np.zeros(hidden_dim)
        self.params['w_fc2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b_fc2'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def saving(self, dir):
        if(not os.path.exists(dir)):
            os.makedirs(dir)
        print("model saving...")
        fp = open(dir + '/' + 'params.txt', 'wb')
        print("saving successfully!")
        pickle.dump(self.params)
        fp.close()

    def restore(self, dir):
        if (os.path.exists(dir)):
            fp = open(dir, 'rb')
            print("model loading...")
            self.params = pickle.load(fp)
            print("loading successfully! ")
            fp.close()
        else:
            raise AssertionError("file does not exist! Please train first.")

    def forward(self, x):
        w_conv1, b_conv1 = self.params['w_conv1'], self.params['b_conv1']
        w_fc1, b_fc1 = self.params['w_fc1'], self.params['b_fc1']
        w_fc2, b_fc2 = self.params['w_fc2'], self.params['b_fc2']

        filter_size = w_conv1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'height': 2, 'width': 2, 'stride': 2}

        #forward
        pool_out1, self.pool_cache1 = conv_relu_pool_forward(x, w_conv1, b_conv1, conv_param, pool_param)

        fc_out1, self.fc_cache1 = fc_relu_forward(pool_out1, w_fc1, b_fc1)
        fc_out2, self.fc_cache2 = fc_forward(fc_out1, w_fc2, b_fc2)
        # Change: add softmax in forward
        prob = softmax_forward(fc_out2)

        return prob, fc_out2

    def iteration(self, x, y):
        #forward
        _, fc_out2 = self.forward(x)
        #backward
        loss, grads = 0, {}
        loss, dout = softmax_loss(fc_out2, y)
        dx_fc2, grads['w_fc2'], grads['b_fc2'] = fc_backward(dout, self.fc_cache2)
        dx_fc1, grads['w_fc1'], grads['b_fc1'] = fc_relu_backward(dx_fc2, self.fc_cache1)
        dx_conv1, grads['w_conv1'], grads['b_conv1'] = conv_relu_pool_backward(dx_fc1, self.pool_cache1)

        return loss, grads

    def loss(self, x, y=None):
        w_conv1, b_conv1 = self.params['w_conv1'], self.params['b_conv1']
        w_fc1, b_fc1 = self.params['w_fc1'], self.params['b_fc1']
        w_fc2, b_fc2 = self.params['w_fc2'], self.params['b_fc2']

        filter_size = w_conv1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'height': 2, 'width': 2, 'stride': 2}

        #forward
        pool_out1, pool_cache1 = conv_relu_pool_forward(x, w_conv1, b_conv1, conv_param, pool_param)

        fc_out1, fc_cache1 = fc_relu_forward(pool_out1, w_fc1, b_fc1)
        fc_out2, fc_cache2 = fc_forward(fc_out1, w_fc2, b_fc2)

        if y is None:
            return softmax_forward(fc_out2)
        
        #backward
        loss, grads = 0, {}
        loss, dout = softmax_loss(fc_out2, y)
        dx_fc2, grads['w_fc2'], grads['b_fc2'] = fc_backward(dout, fc_cache2)
        dx_fc1, grads['w_fc1'], grads['b_fc1'] = fc_relu_backward(dx_fc2, fc_cache1)
        dx_conv1, grads['w_conv1'], grads['b_conv1'] = conv_relu_pool_backward(dx_fc1, pool_cache1)

        return loss, grads




