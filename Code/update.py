import numpy as np


def adam(x, dx, config=None):

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    config['t'] += 1
    # Update biased first moment estimate
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    # Update biased second raw moment estimate
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2) 
    # Compute the correction factor
    alpha = config['learning_rate'] * np.sqrt(1 - config['beta2'] ** config['t']) / (1 - config['beta1'] ** config['t'])
    # Update parameters
    epsilon_corrected = config['epsilon'] * np.sqrt(1 - config['beta2'] ** config['t'])
    next_x = x - alpha * config['m'] / (np.sqrt(config['v']) + epsilon_corrected) 

    return next_x, config