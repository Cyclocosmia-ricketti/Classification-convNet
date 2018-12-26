import numpy as np

def adam(x, dx, adam_params=None):

    if adam_params is None: adam_params = {}
    adam_params.setdefault('learning_rate', 1e-3)
    adam_params.setdefault('beta1', 0.9)
    adam_params.setdefault('beta2', 0.999)
    adam_params.setdefault('epsilon', 1e-8)
    adam_params.setdefault('m', np.zeros_like(x))
    adam_params.setdefault('v', np.zeros_like(x))
    adam_params.setdefault('t', 1)

    adam_params['t'] += 1
    #Update biased first moment estimate
    adam_params['m'] = adam_params['beta1'] * adam_params['m'] + (1 - adam_params['beta1']) * dx
    #Update biased second raw moment estimate 
    adam_params['v'] = adam_params['beta2'] * adam_params['v'] + (1 - adam_params['beta2']) * (dx ** 2) 
    #Compute the correction factor
    alpha = adam_params['learning_rate'] * np.sqrt(1 - adam_params['beta2'] ** adam_params['t']) / (1 - adam_params['beta1'] ** adam_params['t'])
    #Update parameters
    epsilon_corrected = adam_params['epsilon'] * np.sqrt(1 - adam_params['beta2'] ** adam_params['t'])
    x_updated = x - alpha * adam_params['m'] / (np.sqrt(adam_params['v']) + epsilon_corrected) 


    return x_updated, adam_params