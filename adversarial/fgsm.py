import torch


def fgsm(model_fn, x, eps, loss_fn=None, clip_min=-1, clip_max=1, y=None, targeted=False):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        _, y = torch.max(model_fn(x), 1)

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)

    if targeted:
        loss = -loss

    loss.backward()
    perturbation = (eps / 255.0) * torch.sign(x.grad)
    adv_x = x + perturbation
    if (clip_min is not None) or (clip_max is not None):
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x
