import torch
import numpy as np
from fgm import fgm


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError('norm must be np.inf, 1, or 2.')

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            # norm = torch.max(
            #     avoid_zero_div,
            #     torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            # )
        elif norm == 2:
            norm = torch.sqrt(torch.max(
                avoid_zero_div,
                torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
            ))
        factor = torch.min(
            torch.tensor(1., dtype=eta.dtype, device=eta.device),
            eps / norm
        )
        eta = eta * factor
    return eta


def pgd(model_fn, x, eps, eps_iter, nb_iter, norm, loss_fn=None,
        clip_min=None, clip_max=None, y=None, targeted=False,
        rand_init=True, rand_minmax=None):
    if eps == 0:
        return x
    if eps_iter == 0:
        return x

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        # eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
        eta = torch.nn.init.uniform_(torch.empty_like(x), -rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fgm(model_fn, adv_x, eps_iter, norm, loss_fn=loss_fn,
                    clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted)

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    return adv_x
