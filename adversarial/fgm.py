import torch
from cleverhans.torch.utils import optimize_linear


def fgm(model_fn, x, eps, norm, loss_fn=None,
                         clip_min=None, clip_max=None, y=None, targeted=False):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        _, y = torch.max(model_fn(x), 1)

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)

    if targeted:
        loss = -loss

    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)
    adv_x = x + optimal_perturbation
    if (clip_min is not None) or (clip_max is not None):
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x
