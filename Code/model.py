# coding = utf-8

import numpy as np
from layers import *
import pickle
import os


class cnn_VGGLike(object):
    def __init__(self, batch_size=50, input_dim=(3, 64, 64), num_filters=(32, 64, 128, 128, 256, 256),
                 filter_size=(3, 3, 3, 3, 3, 3), hidden_dim=256, num_classes=21, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
            
        self.batch_size = batch_size
        C, H, W = input_dim 

        self.params = {}

        dim1 = 3*64*64
        dim2 = 32*64*64
        dim3 = 32*32*32
        dim4 = 64*32*32
        dim5 = 64*16*16
        dim6 = 128*16*16
        dim7 = 128*16*16
        dim8 = 128*8*8
        dim9 = 256*8*8
        dim10 = 256*8*8
        dim11 = 256*4*4
        dim12 = 256*1
        dim13 = 256*1

        F1, HH1, WW1 = num_filters[0], filter_size[0], filter_size[0]
        low = - np.sqrt(6.0 / (dim1 + dim2))
        high = np.sqrt(6.0 / (dim1 + dim2))
        self.params['w_conv1'] = np.random.uniform(low, high, [F1, C, HH1, WW1])
        self.params['b_conv1'] = np.zeros(F1)

        F2, HH2, WW2 = num_filters[1], filter_size[1], filter_size[1]
        low = - np.sqrt(6.0 / (dim3 + dim4))
        high = np.sqrt(6.0 / (dim3 + dim4))
        self.params['w_conv2'] = np.random.uniform(low, high, [F2, F1, HH2, WW2])
        self.params['b_conv2'] = np.zeros(F2)
        self.params['gamma_bn1'] = np.ones(F2)
        self.params['beta_bn1'] = np.zeros(F2)

        F3, HH3, WW3 = num_filters[2], filter_size[2], filter_size[2]
        low = - np.sqrt(6.0 / (dim5 + dim6))
        high = np.sqrt(6.0 / (dim5 + dim6))
        self.params['w_conv3'] = np.random.uniform(low, high, [F3, F2, HH3, WW3])
        self.params['b_conv3'] = np.zeros(F3)

        F4, HH4, WW4 = num_filters[3], filter_size[3], filter_size[3]
        low = - np.sqrt(6.0 / (dim6 + dim7))
        high = np.sqrt(6.0 / (dim6 + dim7))
        self.params['w_conv4'] = np.random.uniform(low, high, [F4, F3, HH4, WW4])
        self.params['b_conv4'] = np.zeros(F4)
        self.params['gamma_bn2'] = np.ones(F4)
        self.params['beta_bn2'] = np.zeros(F4)

        F5, HH5, WW5 = num_filters[4], filter_size[4], filter_size[4]
        low = - np.sqrt(6.0 / (dim8 + dim9))
        high = np.sqrt(6.0 / (dim8 + dim9))
        self.params['w_conv5'] = np.random.uniform(low, high, [F5, F4, HH5, WW5])
        self.params['b_conv5'] = np.zeros(F5)

        F6, HH6, WW6 = num_filters[5], filter_size[5], filter_size[5]
        low = - np.sqrt(6.0 / (dim9 + dim10))
        high = np.sqrt(6.0 / (dim9 + dim10))
        self.params['w_conv6'] = np.random.uniform(low, high, [F6, F5, HH6, WW6])
        self.params['b_conv6'] = np.zeros(F6)
        self.params['gamma_bn3'] = np.ones(F6)
        self.params['beta_bn3'] = np.zeros(F6)

        low = - np.sqrt(6.0 / (dim11 + dim12))
        high = np.sqrt(6.0 / (dim11 + dim12))
        self.params['w_fc1'] = np.random.uniform(low, high, [dim11, dim12])
        self.params['b_fc1'] = np.zeros(dim12)

        low = - np.sqrt(6.0 / (dim12 + dim13))
        high = np.sqrt(6.0 / (dim12 + dim13))
        self.params['w_fc2'] = np.random.uniform(low, high, [dim12, dim13])
        self.params['b_fc2'] = np.zeros(dim13)

        low = - np.sqrt(6.0 / (dim13 + num_classes))
        high = np.sqrt(6.0 / (dim13 + num_classes))
        self.params['w_fc3'] = np.random.uniform(low, high, [dim13, num_classes])
        self.params['b_fc3'] = np.zeros(num_classes)

        self.bn1_param = {'mode':'train'}
        self.bn2_param = {'mode':'train'}
        self.bn3_param = {'mode':'train'}
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def saving(self, dir, option=-1):
        if(not os.path.exists(dir)):
            os.makedirs(dir)
        print("model saving...")
        if option == -1:
            fp = open(dir + '/' + 'params.txt', 'wb')
        else:
            fp = open(dir + '/' + 'params_%d.txt' % option, 'wb')
        pickle.dump((self.params, self.bn1_param, self.bn2_param, self.bn3_param), fp)
        print("saving successfully!") 
        fp.close()

    def restore(self, dir):
        if (os.path.exists(dir)):
            fp = open(dir, 'rb')
            print("model loading...")
            self.params, self.bn1_param, self.bn2_param, self.bn3_param = pickle.load(fp)
            print("loading successfully! ")
            fp.close()
        else:
            raise AssertionError("file does not exist! Please train first.")

    def forward(self, x, mode='train'):
        w_conv1, b_conv1 = self.params['w_conv1'], self.params['b_conv1']
        w_conv2, b_conv2 = self.params['w_conv2'], self.params['b_conv2']
        w_conv3, b_conv3 = self.params['w_conv3'], self.params['b_conv3']
        w_conv4, b_conv4 = self.params['w_conv4'], self.params['b_conv4']
        w_conv5, b_conv5 = self.params['w_conv5'], self.params['b_conv5']
        w_conv6, b_conv6 = self.params['w_conv6'], self.params['b_conv6']
        w_fc1, b_fc1 = self.params['w_fc1'], self.params['b_fc1']
        w_fc2, b_fc2 = self.params['w_fc2'], self.params['b_fc2']
        w_fc3, b_fc3 = self.params['w_fc3'], self.params['b_fc3']
        gamma_bn1, beta_bn1 = self.params['gamma_bn1'], self.params['beta_bn1']
        gamma_bn2, beta_bn2 = self.params['gamma_bn2'], self.params['beta_bn2']
        gamma_bn3, beta_bn3 = self.params['gamma_bn3'], self.params['beta_bn3']

        conv1_param = {'stride': 1, 'pad': (w_conv1.shape[2] - 1) // 2}
        conv2_param = {'stride': 1, 'pad': (w_conv2.shape[2] - 1) // 2}
        conv3_param = {'stride': 1, 'pad': (w_conv3.shape[2] - 1) // 2}
        conv4_param = {'stride': 1, 'pad': (w_conv4.shape[2] - 1) // 2}
        conv5_param = {'stride': 1, 'pad': (w_conv5.shape[2] - 1) // 2}
        conv6_param = {'stride': 1, 'pad': (w_conv6.shape[2] - 1) // 2}

        pool1_param = {'height': 2, 'width': 2, 'stride': 2}
        pool2_param = {'height': 2, 'width': 2, 'stride': 2}
        pool3_param = {'height': 2, 'width': 2, 'stride': 2}
        pool4_param = {'height': 2, 'width': 2, 'stride': 2}

        bn1_param = self.bn1_param
        bn1_param['mode'] = mode

        bn2_param = self.bn2_param
        bn2_param['mode'] = mode

        bn3_param = self.bn3_param
        bn3_param['mode'] = mode

        #forward
        pool_out1, self.pool_cache1 = conv_relu_pool_forward(x, w_conv1, b_conv1, conv1_param, pool1_param)
        pool_out2, self.pool_cache2 = conv_relu_bn_pool_forward(pool_out1, w_conv2, b_conv2, gamma_bn1, \
                                        beta_bn1, conv2_param, bn1_param, pool2_param)
        conv_out3, self.conv_cache3 = conv_relu_forward(pool_out2, w_conv3, b_conv3, conv3_param)
        pool_out4, self.pool_cache4 = conv_relu_bn_pool_forward(conv_out3, w_conv4, b_conv4, gamma_bn2, \
                                        beta_bn2, conv4_param, bn2_param, pool3_param)
        conv_out5, self.conv_cache5 = conv_relu_forward(pool_out4, w_conv5, b_conv5, conv5_param)
        pool_out6, self.pool_cache6 = conv_relu_bn_pool_forward(conv_out5, w_conv6, b_conv6, gamma_bn3, \
                                        beta_bn3, conv6_param, bn3_param, pool4_param)
        fc_out1, self.fc_cache1 = fc_relu_forward(pool_out6, w_fc1, b_fc1)
        fc_out2, self.fc_cache2 = fc_relu_forward(fc_out1, w_fc2, b_fc2)
        fc_out3, self.fc_cache3 = fc_forward(fc_out2, w_fc3, b_fc3)
        prob = softmax_forward(fc_out3)

        return prob, fc_out3

    def iteration(self, x, y):
        #forward
        _, fc_out2 = self.forward(x)
        #backward
        loss, grads = 0, {}
        loss, dout = softmax_loss(fc_out2, y)
        dx_fc3, grads['w_fc3'], grads['b_fc3'] = fc_backward(dout, self.fc_cache3)
        dx_fc2, grads['w_fc2'], grads['b_fc2'] = fc_relu_backward(dx_fc3, self.fc_cache2)
        dx_fc1, grads['w_fc1'], grads['b_fc1'] = fc_relu_backward(dx_fc2, self.fc_cache1)
        dx_conv6, grads['w_conv6'], grads['b_conv6'], grads['gamma_bn3'], grads['beta_bn3'] = \
                conv_relu_bn_pool_backward(dx_fc1, self.pool_cache6)
        dx_conv5, grads['w_conv5'], grads['b_conv5'] = conv_relu_backward(dx_conv6, self.conv_cache5)
        dx_conv4, grads['w_conv4'], grads['b_conv4'], grads['gamma_bn2'], grads['beta_bn2'] = \
                conv_relu_bn_pool_backward(dx_conv5, self.pool_cache4)
        dx_conv3, grads['w_conv3'], grads['b_conv3'] = conv_relu_backward(dx_conv4, self.conv_cache3)
        dx_conv2, grads['w_conv2'], grads['b_conv2'], grads['gamma_bn1'], grads['beta_bn1'] = \
                conv_relu_bn_pool_backward(dx_conv3, self.pool_cache2)
        dx_conv1, grads['w_conv1'], grads['b_conv1'] = conv_relu_pool_backward(dx_conv2, self.pool_cache1)
        return loss, grads





