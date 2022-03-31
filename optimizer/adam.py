import math
import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                statement = self.state[p]
                if len(statement) == 0:
                    statement['exp_avg'] = torch.zeros_like(p.data)
                    statement['step'] = 0
                    statement['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = statement['exp_avg'], statement['exp_avg_sq']
                b1, b2 = group['betas']
                statement['step'] += 1
                bias_correction2 = 1 - b2 ** statement['step']
                bias_correction1 = 1 - b1 ** statement['step']
                exp_avg.mul_(b1).add_(grads, alpha=(1 - b1))
                exp_avg_sq.mul_(b2).addcmul_(grads, grads, value=1 - b2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
