# coding = utf-8
import numpy as np
from layers import *

class cnn_AlexNet(object):
    def __init__(self, batch_size, input_dim=(3, 64, 64), num_filters=(128), filter_size=(3),
                 hidden_dim=100, num_classes=20, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
            
        self.batch_size = batch_size
        C, H, W = input_dim 
        F1, HH1, WW1 = num_filters[0], filter_size[0], filter_size[0]

        self.fc_cache1 = ()
        self.fc_cache2 = ()
        self.pool_cache1 = ()

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
        pass

    def restore(self, dir):
        pass

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

        return fc_out2
        
    def iteration(self, x, y):
        #forward
        fc_out2 = self.forward(x)
        #backward
        loss, grads = 0, {}
        loss, dout = softmax_loss(fc_out2, y)
        dx_fc2, grads['w_fc2'], grads['b_fc2'] = fc_backward(dout, self.fc_cache2)
        dx_fc1, grads['w_fc1'], grads['b_fc1'] = fc_relu_backward(dx_fc2, self.fc_cache1)
        dx_conv1, grads['w_conv1'], grads['b_conv1'] = conv_relu_pool_backward(dx_fc1, self.pool_cache1)

        return loss, grads




