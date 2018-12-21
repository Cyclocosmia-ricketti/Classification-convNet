import numpy as np

def conv_forward(x, w, b, stride, pad):

    # Get input dimensions
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape

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
    cache = (x, w, b, stride, pad, x_cols)
    return out, cache

def conv_backward(dout, cache):

    # Process cache
    x, w, b, stride, pad, x_cols = cache

    # Calculate db
    db = np.sum(dout, axis=(0, 2, 3))

    # Calculate dw
    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    # Calculate dx
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = col2im(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                       filter_height, filter_width, pad, stride)

    return dx, dw, db

def max_pooling_forward(x, height, width, stride):

    # Get dimensions
    N, C, H, W = x.shape

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
    cache = (x, x_cols, x_cols_argmax, height, width, stride)
    return out, cache

def max_pool_backward_im2col(dout, cache):

    # Process cache
    x, x_cols, x_cols_argmax, height, width, stride = cache

    # Get dimensions
    N, C, H, W = x.shape

    # Calculate dx
    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im_indices(dx_cols, (N * C, 1, H, W), height, width, padding=0, stride=stride)
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
    relu = np.maximum(x, 0)
    dx = np.multiply(dout, np.sign(relu))
    return dx

def fc_forward(x, w, b):

    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
 
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def softmax_forward(x):
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def softmax_backward(dout, cache):
    x = cache
    dx = None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
