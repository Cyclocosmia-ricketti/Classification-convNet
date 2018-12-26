# coding = utf-8
import numpy as np
from .im2col import *


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
    x, x_cols, x_cols_argmax, height, width, stride = cache

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
    N, D = x.shape[0], x.size / x.shape[0]

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
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    # Calculate gradient
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
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
