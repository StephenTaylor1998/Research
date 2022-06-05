import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def local_lip_loss(model, x, xp, top_norm, btm_norm, reduction='mean'):
    # model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    if top_norm == "kl":
        criterion_kl = nn.KLDivLoss(reduction='none')
        top = criterion_kl(F.log_softmax(model(xp), dim=1),
                           F.softmax(model(x), dim=1))
        ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
    else:
        top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
        ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError(f"Not supported reduction: {reduction}")


# def local_lipschitz(model, x, top_norm=1, btm_norm=np.Inf, perturb_steps=10,
#                     step_size=0.003, epsilon=0.01, device="cuda"):
#     x = x.clone().detach().requires_grad_(True)
#     # x_adv = x + 0.001 * torch.randn(x.shape, requires_grad=True).to(device)
#     x_adv = x.clone().detach().requires_grad_(True)
#     for _ in range(perturb_steps):
#
#         # loss = (-1) * local_lip_loss(model, x, x_adv, top_norm, btm_norm, reduction="sum")
#
#         loss = nn.MSELoss()(model(x), model(x_adv))
#         print(loss)
#         loss.backward()
#         print(x.grad.shape)
#         print(x_adv.grad.shape)
#
#         # renorming gradient
#         eta = step_size * x_adv.grad.data.sign().detach()
#         x_adv = x_adv.data.detach() + eta.detach()
#         eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
#         x_adv = x.data.detach() + eta.detach()
#         x_adv = torch.clamp(x_adv, 0, 1.0)
#     return x_adv

import torch.optim as optim


def local_lipschitz(model, x, top_norm, btm_norm, perturb_steps=10,
                          step_size=0.003, epsilon=0.01, device="cuda", print_eta=False):

    x_adv = x + 0.001 * torch.randn(x.shape).to(device)

    # Setup optimizers
    optimizer = optim.SGD([x_adv], lr=step_size)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        optimizer.zero_grad()
        with torch.enable_grad():
            loss = (-1) * local_lip_loss(model, x, x_adv, top_norm, btm_norm)
        loss.backward()
        # renorming gradient
        eta = step_size * x_adv.grad.data.sign().detach()
        x_adv = x_adv.data.detach() + eta.detach()
        eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
        if print_eta: print(eta)
        x_adv = x.data.detach() + eta.detach()
        x_adv = torch.clamp(x_adv, 0, 1.0)
    return x_adv
