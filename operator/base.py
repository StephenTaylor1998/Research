import torch
import numpy as np


def np_one_hot(array, num_classes):
    return np.eye(num_classes, dtype=int)[np.array(array, dtype=int)]


def one_hot(array, num_classes):
    return torch.eye(num_classes, dtype=torch.int)[array]
