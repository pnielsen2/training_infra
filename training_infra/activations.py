import torch
import torch.nn.functional as F
import random

# output_activations

def circular_activation(x):
    alpha = torch.square(x) + .5
    alpha = torch.clip(alpha,min=.0000001, max=10000000000)
    return {'alpha': alpha}

def log_softmax(x):
    log_p = F.log_softmax(x, dim=-1)
    return {'log_p': log_p}

def exp_activation(x):
    alpha = torch.exp(x)
    alpha = torch.clip(alpha,min=.0000001, max=10000000000)
    return {'alpha': alpha}

output_activation_dict = {'circular_activation': circular_activation,
                          'log_softmax': log_softmax,
                          'exp_activation': exp_activation
                          }


# hidden activations

def residual_htanh(x):
    return F.hardtanh(x) + x

hidden_activation_dict = {'selu': F.selu,
                          'relu': F.relu,
                          'residual_htanh': residual_htanh,
                          'gelu': F.gelu,
                          'tanh': F.tanh,
                          'elu': F.elu,
                          'leaky_relu': F.leaky_relu,
                          'softplus': F.softplus
                          }