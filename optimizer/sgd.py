from torch.optim import Optimizer


class Momentum(Optimizer):
    def __init__(self, params, lr=0.1, momentum=.0):
        defaults = dict(lr=lr, momentum=momentum)
        super(Momentum, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            for p in group['params']:
                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buffer = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buffer = param_state['momentum_buffer']
                    buffer.mul_(momentum).add_(d_p)
                d_p = buffer
                p.data.add_(d_p, alpha=-lr)