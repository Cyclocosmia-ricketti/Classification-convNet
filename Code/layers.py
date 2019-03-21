# coding = utf-8
import numpy as np
from im2col import *


def conv_forward(x, w, b, conv_param):

    # Get input dimensions
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # Calculate output
    x_cols = im2col(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    # Saving cache
    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward(dout, cache):

    # Process cache
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    # Calculate db
    db = np.sum(dout, axis=(0, 2, 3))

    # Calculate dw
    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    # Calculate dx
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = col2im(dx_cols, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]),
                       filter_height, filter_width, pad, stride)

    return dx, dw, db


def max_pooling_forward(x, pool_param):

    # Get dimensions
    N, C, H, W = x.shape
    height, width, stride = pool_param['height'], pool_param['width'], pool_param['stride']
    # Check dimensions
    assert (H - height) % stride == 0, 'Invalid height'
    assert (W - width) % stride == 0, 'Invalid width'

    out_height = (H - height) // stride + 1
    out_width = (W - width) // stride + 1

    # Calculate output
    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, height, width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    # Saving cache
    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward(dout, cache):

    # Process cache
    x, x_cols, x_cols_argmax, pool_param = cache
    height, width, stride = pool_param['height'], pool_param['width'], pool_param['stride']
    # Get dimensions
    N, C, H, W = x.shape

    # Calculate dx
    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im(dx_cols, (N * C, 1, H, W), height, width, padding=0, stride=stride)
    dx = dx.reshape(x.shape)

    return dx


def relu_forward(x):

    # Calculate output
    out = np.maximum(x, 0)

    # Saving cache
    cache = x
    return out, cache


def relu_backward(dout, cache):

    # Process cache
    x = cache

    # Calculate dx
    dx = np.multiply(dout, (x > 0))
    return dx


def fc_forward(x, w, b):

    # Get dimension
    N, D = x.shape[0], x.size // x.shape[0]

    # Calculate output
    out = np.dot(x.reshape(N, D), w) + b

    # Saving cache
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):

    # Process cache
    x, w, b = cache

    # Calculate gradient
    N, D = x.shape[0], w.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(N, D).T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var =  np.var(x, axis=0) 
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        cache = (x, x_norm, gamma, beta, sample_mean, sample_var, eps)
    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None

    x, x_norm, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = x.shape
    dx_norm = dout * gamma
    dsample_var = np.sum(-0.5 * dx_norm * x_norm / (sample_var + eps), axis=0)
    sample_std = np.sqrt(sample_var + eps)
    dsample_mean = np.sum(-dx_norm / sample_std, axis=0) + \
            dsample_var * np.sum(-2.0 / N * (x - sample_mean), axis=0)
    dx = dx_norm / sample_std + dsample_var * 2 / N * (x - sample_mean) + dsample_mean / N
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    out, cache = None, None
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape((N*H*W, C))
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape((N*H*W, C))
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta
def softmax_forward(x):

    # Calculate output
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    out = probs / np.sum(probs, axis=1, keepdims=True)

    return out


def softmax_loss(x, y):

    # Calculate softmax
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)

    # Calculate loss
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), (y-1).transpose(1, 0)[0]])) / N

    # Calculate gradient
    dx = probs.copy()
    dx[np.arange(N), (y-1).transpose(1, 0)[0]] -= 1
    dx /= N

    return loss, dx


# some sandwich layers
def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    d_relu = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward(d_relu, conv_cache)
    return dx, dw, db 


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward(x, w, b, conv_param)
    r, relu_cache = relu_forward(a)
    out, pool_cache = max_pooling_forward(r, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout,cache):
    conv_cache, relu_cache, pool_cache = cache
    d_pool = max_pool_backward(dout,pool_cache)
    d_relu = relu_backward(d_pool, relu_cache)
    dx, dw, db = conv_backward(d_relu, conv_cache)
    return dx, dw, db

def conv_relu_bn_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = conv_forward(x, w, b, conv_param)
    r, relu_cache = relu_forward(a)
    b, bn_cache = spatial_batchnorm_forward(r, gamma, beta, bn_param)
    out, pool_cache = max_pooling_forward(b, pool_param)
    cache = (conv_cache, relu_cache, bn_cache, pool_cache)
    return out, cache


def conv_relu_bn_pool_backward(dout, cache):
    conv_cache, relu_cache, bn_cache, pool_cache = cache
    d_pool = max_pool_backward(dout,pool_cache)
    d_bn, dgamma, dbeta = spatial_batchnorm_backward(d_pool, bn_cache)
    d_relu = relu_backward(d_bn, relu_cache)
    dx, dw, db = conv_backward(d_relu, conv_cache)
    return dx, dw, db, dgamma, dbeta

def fc_relu_forward(x, w, b):
    a, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def fc_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    d_relu = relu_backward(dout, relu_cache)
    dx, dw, db = fc_backward(d_relu, fc_cache)
    return dx, dw, db
