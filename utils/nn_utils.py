import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(layer):
    """ Initialize layer with random weights
    """
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)

def silu(x_input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return x_input * torch.sigmoid(x_input)

# Functions to normalize and unnormalize data
def normalize_obs(obs, statistics):
    return (obs - statistics["ob_mean"]) / statistics["ob_std"]

def normalize_deltas(deltas, statistics):
    return (deltas - statistics["delta_mean"]) / statistics["delta_std"]

def unnormalize_deltas(normalized_deltas, statistics):
    return (normalized_deltas * statistics["delta_std"]) + statistics["delta_mean"]
