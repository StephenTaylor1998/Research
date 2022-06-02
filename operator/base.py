import torch
import numpy as np


def np_one_hot(array, num_classes):
    return np.eye(num_classes, dtype=int)[np.array(array, dtype=int)]


def one_hot(array, num_classes):
    return torch.eye(num_classes, dtype=torch.int)[array]


def np_det_nd(array):
    # support nd array det op.
    return np.reshape(
        np.linalg.det(np.reshape(array, (-1, *array.shape[-2:]))),
        array.shape[:-2]
    )


def det_nd(array):
    # support nd array det op.
    return torch.reshape(
        torch.det(torch.reshape(array, (-1, *array.shape[-2:]))),
        array.shape[:-2]
    )


if __name__ == '__main__':
    # inputs = torch.randn(1, 2, 4, 4)
    inputs = torch.randn((4, 4))
    out = det_nd(inputs)
    print(out)
    print(torch.det(inputs))

    # inputs = np.random.randn(1, 2, 4, 4)
    inputs = np.random.randn(4, 4)
    out = np_det_nd(inputs)
    print(out)
    print(np.linalg.det(inputs))